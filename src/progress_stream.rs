//! Server-sent events (SSE) protocol for following a long-running operation
//! over HTTP, shared by the server and `notectl-remote`.
//!
//! A client opts in with `Accept: text/event-stream`. The server then sends
//! any number of `progress` events followed by exactly one `result` event (the
//! operation's usual JSON response) or `error` event (a message). Keep-alive
//! comments hold the connection open through slow steps.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{HeaderMap, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::{Stream, StreamExt, TryStreamExt, stream};
use notectl_core::operation::Operation;
use notectl_core::progress::{Progress, ProgressSink};
use sse_stream::SseStream;
use tokio::sync::{mpsc, watch};

/// Media type a client sends in `Accept` to request the event stream.
pub const MEDIA_TYPE: &str = "text/event-stream";

const PROGRESS_EVENT: &str = "progress";
const RESULT_EVENT: &str = "result";
const ERROR_EVENT: &str = "error";

/// Minimum spacing between `progress` events.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// Whether the request asks for the event stream.
pub fn requested(headers: &HeaderMap) -> bool {
    headers
        .get_all(header::ACCEPT)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|range| range.split(';').next())
        .any(|media_type| media_type.trim().eq_ignore_ascii_case(MEDIA_TYPE))
}

/// Run `operation` and respond with its event stream.
///
/// The operation is cancelled if the client disconnects, matching what happens
/// to a plain JSON request.
pub fn respond(operation: Arc<dyn Operation>, json: serde_json::Value) -> Response {
    // A watch channel keeps only the newest report, so a fast operation or a
    // slow client never builds up a backlog.
    let (progress_tx, mut progress_rx) = watch::channel(None::<Progress>);
    let (event_tx, event_rx) = mpsc::channel::<Event>(1);

    tokio::spawn(async move {
        let sink: ProgressSink = Arc::new(move |progress| {
            progress_tx.send_replace(Some(progress));
        });
        let work = operation.execute_json_with_progress(json, sink);

        // Forwarding runs until the operation drops its sink, then drains the
        // last report.
        let forward = async {
            while progress_rx.changed().await.is_ok() {
                let latest = progress_rx.borrow_and_update().clone();
                if let Some(progress) = latest {
                    let event = json_event(PROGRESS_EVENT, &progress);
                    if event_tx.send(event).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(PROGRESS_INTERVAL).await;
                }
            }
        };

        let run = async {
            let (result, ()) = tokio::join!(work, forward);
            let last = match result {
                Ok(value) => json_event(RESULT_EVENT, &value),
                Err(e) => Event::default()
                    .event(ERROR_EVENT)
                    .data(format!("Operation failed: {}", e.message)),
            };
            // Fails only if the client has gone, leaving nobody to tell.
            let _ = event_tx.send(last).await;
        };

        tokio::select! {
            () = run => {}
            () = event_tx.closed() => {}
        }
    });

    let events = stream::unfold(event_rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<_, Infallible>(event), rx))
    });

    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn json_event(name: &str, data: &impl serde::Serialize) -> Event {
    // Compact JSON has no newlines, so it always fits one `data:` line.
    let data = serde_json::to_string(data).expect("progress and results serialize to JSON");
    Event::default().event(name).data(data)
}

/// Consume an event stream body, passing reports to `progress`.
///
/// Returns the JSON text of the `result` event.
pub async fn read<S, B, E>(body: S, progress: &ProgressSink) -> Result<String, String>
where
    S: Stream<Item = Result<B, E>>,
    B: Into<axum::body::Bytes>,
    E: std::error::Error + Send + Sync + 'static,
{
    let events = SseStream::from_bytes_stream(body.map_ok(Into::into));
    let mut events = std::pin::pin!(events);

    while let Some(event) = events.next().await {
        let event = event.map_err(|e| format!("Failed to read progress stream: {e}"))?;
        let data = event.data.unwrap_or_default();
        match event.event.as_deref() {
            Some(PROGRESS_EVENT) => {
                let report = serde_json::from_str(&data)
                    .map_err(|e| format!("Invalid progress event: {e}"))?;
                progress(report);
            }
            Some(RESULT_EVENT) => return Ok(data),
            Some(ERROR_EVENT) => return Err(data),
            // Keep-alive comments, and event types added by newer servers.
            _ => {}
        }
    }

    Err("Server closed the connection before sending a result".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use notectl_core::config::Config;
    use notectl_search::{IndexOperation, SearchCapability};
    use std::sync::Mutex;

    fn recording_sink() -> (ProgressSink, Arc<Mutex<Vec<Progress>>>) {
        let reports = Arc::new(Mutex::new(Vec::new()));
        let recorded = reports.clone();
        let sink: ProgressSink = Arc::new(move |p| recorded.lock().unwrap().push(p));
        (sink, reports)
    }

    fn index_operation(vault: &std::path::Path) -> Arc<dyn Operation> {
        Arc::new(IndexOperation::new(Arc::new(SearchCapability::new(
            vault.to_path_buf(),
            Arc::new(Config::default()),
        ))))
    }

    #[test]
    fn requested_matches_the_media_type_within_an_accept_list() {
        let mut headers = HeaderMap::new();
        assert!(!requested(&headers));
        headers.insert(header::ACCEPT, HeaderValue::from_static("*/*"));
        assert!(!requested(&headers));
        headers.insert(
            header::ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream;q=0.9"),
        );
        assert!(requested(&headers));
    }

    #[tokio::test]
    async fn server_stream_round_trips_through_the_client() {
        let vault = tempfile::tempdir().unwrap();
        std::fs::write(vault.path().join("a.md"), "# A\n\nSome note text.").unwrap();

        let response = respond(index_operation(vault.path()), serde_json::json!({}));
        let (sink, reports) = recording_sink();
        let result = read(response.into_body().into_data_stream(), &sink)
            .await
            .unwrap();

        let result: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(result["files_indexed"], 1);
        // The final report survives throttling.
        let last = reports.lock().unwrap().last().cloned().unwrap();
        assert_eq!((last.current, last.total), (1, Some(1)));
    }

    #[tokio::test]
    async fn operation_failure_arrives_as_an_error() {
        let vault = tempfile::tempdir().unwrap();
        let response = respond(
            index_operation(vault.path()),
            serde_json::json!({ "dim": "not a number" }),
        );
        let message = read(
            response.into_body().into_data_stream(),
            &notectl_core::progress::ignore(),
        )
        .await
        .unwrap_err();
        assert!(message.starts_with("Operation failed:"), "{message}");
    }

    #[tokio::test]
    async fn stream_without_a_result_is_an_error() {
        let body = stream::iter([Ok::<_, std::io::Error>(": keep-alive\n\n")]);
        let result = read(body, &notectl_core::progress::ignore()).await;
        assert!(result.unwrap_err().contains("before sending a result"));
    }
}

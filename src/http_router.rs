use axum::{
    Router,
    extract::{Json, Query},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
};
use notectl::progress_stream;
use notectl_core::operation::Operation;
use std::sync::Arc;

/// Register an HTTP operation on a router
///
/// Creates both GET and POST routes for the operation at its specified path.
/// The router state type must remain generic to work with the application's state.
pub fn register_operation<S>(
    router: Router<S>,
    operation: Arc<dyn notectl_core::operation::Operation>,
) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let path = operation.path();
    let op_get = operation.clone();
    let op_post = operation;

    router.route(
        path,
        get({
            move |headers: HeaderMap,
                  Query(params): Query<serde_json::Map<String, serde_json::Value>>| {
                execute(op_get.clone(), headers, serde_json::Value::Object(params))
            }
        })
        .post({
            move |headers: HeaderMap, Json(json_request): Json<serde_json::Value>| {
                execute(op_post.clone(), headers, json_request)
            }
        }),
    )
}

/// Respond with a progress event stream when the client asks for one and the
/// operation has progress to report, otherwise with a single JSON body.
async fn execute(
    operation: Arc<dyn Operation>,
    headers: HeaderMap,
    json_request: serde_json::Value,
) -> Response {
    if operation.reports_progress() && progress_stream::requested(&headers) {
        return progress_stream::respond(operation, json_request);
    }

    match operation.execute_json(json_request).await {
        Ok(json_response) => Json(json_response).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Operation failed: {}", e.message),
        )
            .into_response(),
    }
}

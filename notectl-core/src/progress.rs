//! Progress reporting for long-running operations.
//!
//! Operations describe how far along they are through a [`ProgressSink`] and
//! stay unaware of who is listening: a terminal progress bar for local runs,
//! or an event stream for HTTP clients.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

/// A snapshot of an operation's progress. Each report replaces the previous
/// one, so a listener may drop intermediate reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    /// What the operation is doing right now.
    pub message: String,
    /// Units of work completed.
    pub current: u64,
    /// Total units of work, absent while it is still unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

/// Receiver of progress reports. Must not block.
pub type ProgressSink = Arc<dyn Fn(Progress) + Send + Sync>;

/// A sink that discards every report.
pub fn ignore() -> ProgressSink {
    Arc::new(|_| {})
}

/// A sink that draws a progress bar on stderr.
///
/// Nothing is drawn until the first report, or when stderr is not a terminal.
/// The bar is cleared when the last clone of the sink is dropped.
pub fn terminal() -> ProgressSink {
    struct Bar(OnceLock<ProgressBar>);

    impl Drop for Bar {
        fn drop(&mut self) {
            if let Some(bar) = self.0.get() {
                bar.finish_and_clear();
            }
        }
    }

    let bar = Bar(OnceLock::new());
    Arc::new(move |progress: Progress| {
        // Capture `bar` as a whole so its Drop runs with the closure.
        let holder = &bar;
        let bar = holder.0.get_or_init(|| {
            let bar = ProgressBar::new_spinner();
            // Keeps the spinner and clock moving during slow steps.
            bar.enable_steady_tick(Duration::from_millis(100));
            bar
        });

        let template = match progress.total {
            Some(_) => "{spinner:.green} [{elapsed}] {bar:30} {pos}/{len} {msg}",
            None => "{spinner:.green} [{elapsed}] {msg}",
        };
        bar.set_style(
            ProgressStyle::with_template(template)
                .expect("progress template is valid")
                .progress_chars("##-"),
        );
        if let Some(total) = progress.total {
            bar.set_length(total);
        }
        bar.set_position(progress.current);
        bar.set_message(progress.message);
    })
}

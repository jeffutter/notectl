pub mod config;
pub mod error;
pub mod file_walker;
pub mod operation;

pub use config::Config;
pub use error::{internal_error, invalid_params, json_error};
pub use operation::Operation;

/// Result type for capability operations
pub type CapabilityResult<T> = Result<T, rmcp::model::ErrorData>;

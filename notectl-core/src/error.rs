use rmcp::model::{ErrorCode, ErrorData};
use std::borrow::Cow;

/// Creates a JSON-RPC error with a custom error code and message
pub fn json_error(code: i32, msg: impl Into<String>) -> ErrorData {
    ErrorData {
        code: ErrorCode(code),
        message: Cow::from(msg.into()),
        data: None,
    }
}

/// Creates a JSON-RPC internal error (-32603)
/// Use for server-side errors like I/O failures, parsing errors, etc.
pub fn internal_error(msg: impl Into<String>) -> ErrorData {
    json_error(-32603, msg)
}

/// Creates a JSON-RPC invalid params error (-32602)
/// Use for invalid input parameters, paths outside bounds, etc.
pub fn invalid_params(msg: impl Into<String>) -> ErrorData {
    json_error(-32602, msg)
}

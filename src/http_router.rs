use axum::{
    Router,
    extract::{Json, Query},
    routing::get,
};
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
            move |Query(params): Query<serde_json::Map<String, serde_json::Value>>| {
                let op = op_get.clone();
                async move {
                    let json_request = serde_json::Value::Object(params);
                    let json_response = op.execute_json(json_request).await.map_err(|e| {
                        (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Operation failed: {}", e.message),
                        )
                    })?;
                    Ok::<_, (axum::http::StatusCode, String)>(Json(json_response))
                }
            }
        })
        .post({
            move |Json(json_request): Json<serde_json::Value>| {
                let op = op_post.clone();
                async move {
                    let json_response = op.execute_json(json_request).await.map_err(|e| {
                        (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Operation failed: {}", e.message),
                        )
                    })?;
                    Ok::<_, (axum::http::StatusCode, String)>(Json(json_response))
                }
            }
        }),
    )
}

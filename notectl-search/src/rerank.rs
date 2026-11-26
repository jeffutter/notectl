//! Relevance scoring through an HTTP `/rerank` endpoint.
//!
//! A reranker reads the query together with each candidate document and scores
//! the pair, so its scores set both the order of results and how relevant they
//! are. The request and response follow the Cohere-style shape that LiteLLM,
//! vLLM and llama.cpp serve.

use notectl_core::config::SearchConfig;

/// Client for `POST {embedding_api_base}/rerank`.
pub(crate) struct Reranker<'a> {
    api_base: &'a str,
    api_key: Option<&'a str>,
    model: &'a str,
}

#[derive(serde::Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
}

#[derive(serde::Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(serde::Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f64,
}

impl<'a> Reranker<'a> {
    /// `None` unless both a rerank model and an API base URL are configured.
    pub fn from_search_config(config: &'a SearchConfig) -> Option<Self> {
        Some(Self {
            api_base: config.embedding_api_base.as_deref()?,
            api_key: config.embedding_api_key.as_deref(),
            model: config.rerank_model.as_deref()?,
        })
    }

    /// The relevance of each document to `query`, from 0 to 1, in the order of
    /// `documents`. Anything short of a usable score for every document is an
    /// error.
    pub async fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f64>, String> {
        let url = format!("{}/rerank", self.api_base.trim_end_matches('/'));
        // Long enough for a server that has to load the model first.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {e}"))?;
        let mut request = client.post(&url).json(&RerankRequest {
            model: self.model,
            query,
            documents,
        });
        if let Some(key) = self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("Failed to reach {url}: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("{url} returned {status}: {body}"));
        }
        let parsed: RerankResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response from {url}: {e}"))?;

        in_document_order(parsed.results, documents.len())
    }
}

fn in_document_order(results: Vec<RerankResult>, count: usize) -> Result<Vec<f64>, String> {
    let mut scores = vec![None; count];
    for result in results {
        if !(0.0..=1.0).contains(&result.relevance_score) {
            return Err(format!(
                "Rerank score {} is outside 0..1",
                result.relevance_score
            ));
        }
        let slot = scores
            .get_mut(result.index)
            .ok_or_else(|| format!("Rerank result for unknown document {}", result.index))?;
        *slot = Some(result.relevance_score);
    }
    scores
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| "Rerank response did not score every document".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(index: usize, relevance_score: f64) -> RerankResult {
        RerankResult {
            index,
            relevance_score,
        }
    }

    #[test]
    fn scores_follow_document_order() {
        let scores = in_document_order(vec![result(1, 0.9), result(0, 0.1)], 2);
        assert_eq!(scores, Ok(vec![0.1, 0.9]));
    }

    #[test]
    fn unusable_responses_are_errors() {
        assert!(
            in_document_order(vec![result(0, 0.5)], 2).is_err(),
            "missing"
        );
        assert!(
            in_document_order(vec![result(1, 0.5)], 1).is_err(),
            "unknown"
        );
        assert!(in_document_order(vec![result(0, 4.2)], 1).is_err(), "range");
    }

    #[test]
    fn needs_both_a_model_and_an_endpoint() {
        let mut config = SearchConfig {
            rerank_model: Some("rerank".to_string()),
            ..Default::default()
        };
        assert!(Reranker::from_search_config(&config).is_none());
        config.embedding_api_base = Some("http://localhost/v1".to_string());
        assert!(Reranker::from_search_config(&config).is_some());
    }
}

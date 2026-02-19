//! SQLite-based vector store for Rig.
//!
//! Stores documents as JSON with embeddings in a `sqlite-vec` virtual table,
//! following the same pattern as `rig-postgres` and `rig-mongodb`.
//!
//! # Example
//! ```rust,no_run
//! use rig::{
//!     Embed,
//!     embeddings::EmbeddingsBuilder,
//!     providers::openai::{self, Client},
//!     vector_store::{VectorStoreIndex, InsertDocuments},
//!     vector_store::request::VectorSearchRequest,
//!     client::{EmbeddingsClient, ProviderClient},
//! };
//! use rig_sqlite::SqliteVectorStore;
//! use serde::{Deserialize, Serialize};
//! use tokio_rusqlite::Connection;
//!
//! #[derive(Embed, Clone, Debug, Serialize, Deserialize)]
//! struct Document {
//!     id: String,
//!     #[embed]
//!     content: String,
//! }
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let conn = Connection::open("vector_store.db").await?;
//! let openai_client = Client::from_env();
//! let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
//!
//! let vector_store = SqliteVectorStore::new(conn, model.clone(), "documents").await?;
//!
//! let documents = vec![
//!     Document { id: "doc1".into(), content: "Hello world".into() },
//!     Document { id: "doc2".into(), content: "Goodbye world".into() },
//! ];
//!
//! let embeddings = EmbeddingsBuilder::new(model)
//!     .documents(documents)?
//!     .build()
//!     .await?;
//!
//! vector_store.insert_documents(embeddings).await?;
//!
//! let results = vector_store
//!     .top_n::<Document>(VectorSearchRequest::builder().query("hello").samples(1).build()?)
//!     .await?;
//! # Ok(())
//! # }
//! ```

use std::ops::RangeInclusive;

use rig::Embed;
use rig::OneOrMany;
use rig::embeddings::{Embedding, EmbeddingModel};
use rig::vector_store::request::{FilterError, SearchFilter, VectorSearchRequest};
use rig::vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex};
use rig::wasm_compat::WasmCompatSend;
use rusqlite::types::Value;
use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;
use tracing::debug;
use zerocopy::IntoBytes;

/// SQLite-backed vector store that persists documents as JSON.
///
/// Uses `sqlite-vec` for vector similarity search. Each document is stored
/// as a JSON string alongside its embedding vectors. The embedding model is
/// stored alongside the connection so the same model is always used for both
/// inserts and queries.
#[derive(Clone)]
pub struct SqliteVectorStore<Model: EmbeddingModel> {
    model: Model,
    conn: Connection,
    table_name: String,
}

impl<Model> SqliteVectorStore<Model>
where
    Model: EmbeddingModel,
{
    /// Create a new vector store, initialising the document and embedding tables
    /// if they do not already exist.
    pub async fn new(
        model: Model,
        conn: Connection,
        table_name: impl Into<String>,
    ) -> Result<Self, VectorStoreError> {
        let dims = model.ndims();
        let table_name = table_name.into();

        let tn = table_name.clone();
        conn.call(move |conn| {
            conn.execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS {tn} (\
                     id TEXT PRIMARY KEY,\
                     document TEXT NOT NULL,\
                     embedded_text TEXT NOT NULL\
                 )"
            ))?;

            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS {tn}_embeddings \
                 USING vec0(embedding float[{dims}])"
            ))?;

            Ok(())
        })
        .await
        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(Self {
            conn,
            model,
            table_name,
        })
    }
}

impl<Model> InsertDocuments for SqliteVectorStore<Model>
where
    Model: EmbeddingModel,
{
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let table_name = self.table_name.clone();

        // Pre-serialise documents so everything is WasmCompatSend + 'static
        // for the closure passed to `Connection::call`.
        let rows = documents
            .into_iter()
            .map(|(doc, embeddings)| {
                let json = serde_json::to_string(&doc)?;
                Ok((json, embeddings))
            })
            .collect::<Result<Vec<_>, serde_json::Error>>()?;

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                {
                    let mut doc_stmt = tx.prepare(&format!(
                        "INSERT OR REPLACE INTO {table_name} (id, document, embedded_text) \
                         VALUES (?1, ?2, ?3)"
                    ))?;

                    let mut emb_stmt = tx.prepare(&format!(
                        "INSERT INTO {table_name}_embeddings (rowid, embedding) \
                         VALUES (?1, ?2)"
                    ))?;

                    for (json, embeddings) in &rows {
                        for embedding in embeddings.iter() {
                            doc_stmt.execute(rusqlite::params![
                                &embedding.document,
                                json,
                                &embedding.document,
                            ])?;
                            let rowid = tx.last_insert_rowid();

                            let vec = serialize_embedding(embedding);
                            let blob = Value::Blob(vec.as_bytes().to_vec());
                            emb_stmt.execute(rusqlite::params![rowid, blob])?;
                        }
                    }
                }

                tx.commit()?;
                Ok(())
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }
}

#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct SqliteSearchFilter {
    condition: String,
    params: Vec<serde_json::Value>,
}

impl SearchFilter for SqliteSearchFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            condition: format!("{} = ?", key.as_ref()),
            params: vec![value],
        }
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            condition: format!("{} > ?", key.as_ref()),
            params: vec![value],
        }
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            condition: format!("{} < ?", key.as_ref()),
            params: vec![value],
        }
    }

    fn and(self, rhs: Self) -> Self {
        Self {
            condition: format!("({}) AND ({})", self.condition, rhs.condition),
            params: self.params.into_iter().chain(rhs.params).collect(),
        }
    }

    fn or(self, rhs: Self) -> Self {
        Self {
            condition: format!("({}) OR ({})", self.condition, rhs.condition),
            params: self.params.into_iter().chain(rhs.params).collect(),
        }
    }
}

impl SqliteSearchFilter {
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self {
            condition: format!("NOT ({})", self.condition),
            ..self
        }
    }

    /// Tests whether the value at `key` is contained in the range
    pub fn between<N>(key: String, range: RangeInclusive<N>) -> Self
    where
        N: Ord + rusqlite::ToSql + std::fmt::Display,
    {
        let lo = range.start();
        let hi = range.end();

        Self {
            condition: format!("{key} between {lo} and {hi}"),
            ..Default::default()
        }
    }

    // Null checks
    pub fn is_null(key: String) -> Self {
        Self {
            condition: format!("{key} is null"),
            ..Default::default()
        }
    }

    pub fn is_not_null(key: String) -> Self {
        Self {
            condition: format!("{key} is not null"),
            ..Default::default()
        }
    }

    // String ops
    /// Tests whether the value at `key` satisfies the glob pattern
    /// `pattern` should be a valid SQLite glob pattern
    pub fn glob<'a, S>(key: String, pattern: S) -> Self
    where
        S: AsRef<&'a str>,
    {
        Self {
            condition: format!("{key} glob {}", pattern.as_ref()),
            ..Default::default()
        }
    }

    /// Tests whether the value at `key` satisfies the "like" pattern
    /// `pattern` should be a valid SQLite like pattern
    pub fn like<'a, S>(key: String, pattern: S) -> Self
    where
        S: AsRef<&'a str>,
    {
        Self {
            condition: format!("{key} like {}", pattern.as_ref()),
            ..Default::default()
        }
    }

    fn compile_params(self) -> Result<Vec<Value>, FilterError> {
        self.params
            .into_iter()
            .map(convert_json_to_sqlite)
            .collect()
    }
}

fn convert_json_to_sqlite(value: serde_json::Value) -> Result<Value, FilterError> {
    match value {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(b) => Ok(Value::Integer(b as i64)),
        serde_json::Value::String(s) => Ok(Value::Text(s)),
        serde_json::Value::Number(n) => {
            if let Some(float) = n.as_f64() {
                Ok(Value::Real(float))
            } else if let Some(int) = n.as_i64() {
                Ok(Value::Integer(int))
            } else {
                Err(FilterError::Serialization(
                    "unsupported numeric type".to_string(),
                ))
            }
        }
        serde_json::Value::Array(arr) => {
            let blob =
                serde_json::to_vec(&arr).map_err(|e| FilterError::Serialization(e.to_string()))?;
            Ok(Value::Blob(blob))
        }
        serde_json::Value::Object(obj) => {
            let blob =
                serde_json::to_vec(&obj).map_err(|e| FilterError::Serialization(e.to_string()))?;
            Ok(Value::Blob(blob))
        }
    }
}

fn build_where_clause(
    req: &VectorSearchRequest<SqliteSearchFilter>,
    query_vec: Vec<f32>,
) -> Result<(String, Vec<Value>), FilterError> {
    let thresh = req.threshold().unwrap_or(0.);
    let thresh = SqliteSearchFilter::gt("distance", thresh.into());

    let filter = req
        .filter()
        .as_ref()
        .cloned()
        .map(|filter| thresh.clone().and(filter))
        .unwrap_or(thresh);

    let where_clause = format!(
        "WHERE e.embedding MATCH ? AND k = ? AND {}",
        filter.condition
    );

    let query_vec = query_vec.into_iter().flat_map(f32::to_le_bytes).collect();
    let query_vec = Value::Blob(query_vec);
    let samples = req.samples() as u32;

    let mut params = vec![query_vec.clone(), query_vec, samples.into()];
    let filter_params = filter.clone().compile_params()?;
    params.extend(filter_params);

    Ok((where_clause, params))
}

impl<Model> VectorStoreIndex for SqliteVectorStore<Model>
where
    Model: EmbeddingModel,
{
    type Filter = SqliteSearchFilter;

    async fn top_n<D>(
        &self,
        req: VectorSearchRequest<SqliteSearchFilter>,
    ) -> Result<Vec<(f64, String, D)>, VectorStoreError>
    where
        D: for<'de> Deserialize<'de>,
    {
        let embedding = self.model.embed_text(req.query()).await?;
        let query_vec = serialize_embedding(&embedding);
        let table_name = self.table_name.clone();

        let (where_clause, params) = build_where_clause(&req, query_vec)?;

        let rows = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "SELECT d.id, d.document, \
                            (1 - vec_distance_cosine(?, e.embedding)) AS distance \
                     FROM {table_name}_embeddings e \
                     JOIN {table_name} d ON e.rowid = d.rowid \
                     {where_clause} \
                     ORDER BY distance"
                ))?;

                let rows = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
                        let id: String = row.get(0)?;
                        let document: String = row.get(1)?;
                        let distance: f64 = row.get(2)?;
                        Ok((id, document, distance))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let mut results = Vec::with_capacity(rows.len());
        for (id, doc_json, distance) in rows {
            match serde_json::from_str::<D>(&doc_json) {
                Ok(doc) => results.push((distance, id, doc)),
                Err(e) => {
                    debug!("Failed to deserialize document {id}: {e}");
                    continue;
                }
            }
        }

        Ok(results)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<SqliteSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = self.model.embed_text(req.query()).await?;
        let query_vec = serialize_embedding(&embedding);
        let table_name = self.table_name.clone();

        let (where_clause, params) = build_where_clause(&req, query_vec)?;

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "SELECT d.id, \
                            (1 - vec_distance_cosine(?1, e.embedding)) AS distance \
                     FROM {table_name}_embeddings e \
                     JOIN {table_name} d ON e.rowid = d.rowid \
                     {where_clause} \
                     ORDER BY distance"
                ))?;

                let results = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
                        Ok((row.get::<_, f64>(1)?, row.get::<_, String>(0)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(results)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }
}

fn serialize_embedding(embedding: &Embedding) -> Vec<f32> {
    embedding.vec.iter().map(|x| *x as f32).collect()
}

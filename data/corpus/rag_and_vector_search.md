# Retrieval-Augmented Generation and Vector Search

Retrieval-Augmented Generation (RAG) is a technique that grounds a language model's
answers in an external knowledge base. Instead of relying only on what the model learned
during training, a RAG system retrieves relevant passages at query time and provides them
to the model as context. This reduces hallucination, allows the use of private or
up-to-date data, and lets answers cite their sources.

## The RAG Pipeline

A typical RAG system has two phases:

1. **Ingestion (offline)**: Documents are loaded, split into chunks, embedded into
   vectors, and stored in a vector database.
2. **Query (online)**: The user's question is embedded, the most similar chunks are
   retrieved, and a language model generates an answer using those chunks as context.

## Chunking

Documents are split into smaller chunks because embedding models have a limited context
window and because retrieval is more precise when each unit of text covers a single idea.
Key parameters are:

- **Chunk size**: The maximum length of each chunk, measured in characters or tokens.
  Larger chunks preserve more context but dilute relevance; smaller chunks are more
  precise but may lose surrounding meaning. A common starting range is 500 to 1000 tokens.
- **Chunk overlap**: A number of characters repeated between consecutive chunks so that a
  sentence split across a boundary still appears intact in at least one chunk. A typical
  overlap is 10 to 20 percent of the chunk size.

For structured technical content, splitting along semantic boundaries such as Markdown
headers keeps related material together and preserves the section a chunk belongs to as
metadata.

## Embeddings

An embedding model converts text into a fixed-length vector of floating-point numbers.
Texts with similar meaning map to vectors that are close together in the vector space.
The similarity between two vectors is commonly measured with cosine similarity, which
compares the angle between them regardless of their magnitude.

The dimensionality of the vector depends on the model. For example, the
`all-MiniLM-L6-v2` sentence-transformer produces 384-dimensional vectors and is small,
fast, and effective for general-purpose retrieval, which makes it a good default for
local prototyping that does not require an API key.

## Vector Stores

A vector store indexes embeddings so that nearest-neighbour search is fast even over
millions of vectors. Common choices include:

- **ChromaDB**: An open-source vector database that runs locally, persists to disk, and is
  simple to set up, making it well suited to prototyping.
- **FAISS**: A high-performance similarity-search library from Facebook AI. It is very
  fast but is a library rather than a full database, so metadata handling is more manual.

A similarity search returns the top-k chunks whose embeddings are closest to the query
embedding, along with their stored metadata such as the source document and section.

## Self-Corrective RAG

Basic RAG retrieves and generates in a single pass, which fails when retrieval returns
irrelevant chunks. Self-corrective variants add checks:

- **Corrective RAG (CRAG)**: Grades each retrieved chunk for relevance. If the chunks are
  poor, it rewrites the query and retrieves again, or falls back to web search.
- **Self-RAG**: Adds reflection steps that check whether the generated answer is actually
  supported by the retrieved context (a groundedness or hallucination check) before
  returning it.

These patterns map naturally onto a graph: retrieval, grading, and generation become
nodes, and the decision to retry or proceed becomes a conditional edge.

## Grading and Hallucination Checks

A document grader uses a language model to label each retrieved chunk as relevant or
irrelevant to the question, filtering out noise before generation. A hallucination check
(also called a groundedness check) asks the model whether the final answer is entailed by
the retrieved context; if it is not, the system can regenerate, retrieve more context, or
warn the user. Together these checks make the pipeline more trustworthy at the cost of
extra model calls and latency.

## Citations

Because a RAG system knows which chunks it used, it can attach citations to the answer by
referencing the source metadata of each chunk. Citations let users verify claims and
distinguish grounded statements from the model's own prior knowledge.

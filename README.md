Name: Apryl Essilfua Poku
Roll Number: 10012300025

I chose a chunk size of 500 words because it provides enough context for the LLM to understand financial policies in the Budget PDF without exceeding the token limits of standard embedding models. The 50-word overlap ensures that critical sentences or data points that fall on the boundary of a chunk are not abruptly cut in half, preserving semantic meaning during retrieval.

```mermaid
flowchart TD
    subgraph Offline [Data Ingestion Pipeline]
        Doc1(Ghana Election CSV) --> Ext1(Data Extraction: Pandas)
        Doc2(2025 Budget PDF) --> Ext2(Data Extraction: PyPDF2)
        Ext1 --> Chunk(Manual Sliding Window Chunker)
        Ext2 --> Chunk
        Chunk --> Embed1(Embedding Model: SentenceTransformers)
        Embed1 --> FAISS[(Vector Database: FAISS IndexFlatL2)]
    end

    subgraph Online [Retrieval & Generation Pipeline]
        UI[Streamlit UI] --> Query(User Query)
        Query --> Embed2(Query Embedding)
        Embed2 --> Search(Top-K Similarity Search)
        Search --> FAISS
        FAISS --> Context(Retrieved Context Chunks)
        Query --> Prompt{Prompt Constructor & Context Injection}
        Context --> Prompt
        Prompt --> LLM(Generative LLM API)
        LLM --> Output[Final Response]
        Output --> UI
    end
```
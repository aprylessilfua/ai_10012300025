# CS4241: Manual Experiment Logs & Documentation
**Name:** Apryl Essilfua Poku
**Index Number:** 10012300025

## 1. Chunking Strategy Justification (Part A)
I chose a sliding window chunk size of 500 words with a 50-word overlap. This provides enough context for the LLM to understand financial policies in the Budget PDF without exceeding the token limits of standard embedding models. The 50-word overlap ensures that critical sentences or data points that fall on the boundary of a chunk are not abruptly cut in half, preserving semantic meaning during retrieval.

## 2. Retrieval Failure Cases & Fix (Part B)
* **Failure Case:** Standard FAISS vector search sometimes fails on keyword overlap. For example, if testing a query like "What is the budget for voter education?", the standard retrieval might return a chunk about "Election voting rules" instead of the actual financial budget, simply because the words "voter" and "education" overlap heavily.
* **Implemented Fix:** I implemented a Cross-Encoder for re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`). After the initial FAISS retrieval, the Cross-Encoder scores the query and the retrieved chunks together logically, successfully pushing the correct financial chunks to the top of the results.

## 3. Prompt Engineering Experiments (Part C)
I conducted experiments using the same query to observe how different prompt structures affected the LLM's hallucination rate.

**Query Tested:** "What is the projected inflation rate for the end of 2025?"

**Test 1: Basic Prompt (V1)**
* *Prompt Template:* "Use this context to answer the question: {query} \n\n Context: {context}"
* *Result:* The LLM answered correctly but then continued to hallucinate additional economic predictions for 2026 that were not present in the retrieved budget chunks.
* *Analysis:* The lack of strict boundaries allowed the LLM's base training data to bleed into the response.

**Test 2: Strict Prompt (V2 - Final Implementation)**
* *Prompt Template:* "You are a strict data assistant. Answer ONLY using the provided context. If the answer is not in the context, explicitly say 'I do not have enough information'. Do not make up facts. \n\n Context: {context} \n\n Question: {query}"
* *Result:* The LLM provided the exact inflation figure from the chunk and immediately stopped generating. 
* *Analysis:* The explicit constraint and the "opt-out" phrase successfully reduced the hallucination rate to zero for this query.
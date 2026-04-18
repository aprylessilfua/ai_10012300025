# Name: Apryl Essilfua Poku
# Index Number: 10012300025
# CS4241: Streamlit front-end for the custom RAG pipeline

from __future__ import annotations

import importlib.util
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent


def _load_module(qualname: str, filename: str):
    path = _ROOT / filename
    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="Loading models and indexing PDF…")
def _retriever_and_run_fn():
    """Load ``4_pipeline`` / ``1_data_prep``, index PDF chunks once, return retriever + ``run_rag_pipeline``."""
    pipeline = _load_module("rag_pipeline", "4_pipeline.py")
    data_prep = _load_module("data_prep_main", "1_data_prep.py")
    if not data_prep.PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {data_prep.PDF_PATH}")
    pdf_text = data_prep.extract_text_from_pdf(data_prep.PDF_PATH)
    pdf_chunks = data_prep.sliding_window_chunks(pdf_text)
    retriever = pipeline.FaissChunkRetriever()
    retriever.index_chunks(pdf_chunks)
    return retriever, pipeline.run_rag_pipeline


st.set_page_config(page_title="Academic City RAG Assistant", layout="wide")

st.title("Academic City RAG Assistant")

if not Path(__file__).resolve().parent.joinpath("data").exists():
    st.warning("Expected a `data` folder next to `app.py`.")

with st.form("query_form"):
    user_query = st.text_input("Your question", placeholder="Ask about the indexed budget document…")
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            retriever, run_rag_pipeline = _retriever_and_run_fn()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            st.stop()

        with st.spinner("Running retrieval, re-ranking, and Gemini…"):
            try:
                result = run_rag_pipeline(user_query.strip(), retriever)
            except RuntimeError as e:
                st.error(str(e))
                st.caption("Set `GEMINI_API_KEY` in your environment before running queries.")
                st.stop()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        st.subheader("Final LLM Response")
        st.markdown(result.get("answer") or "_No answer text returned._")

        hits = result.get("retrieval_hits") or []
        reranked = result.get("reranked") or []

        with st.expander("Retrieved chunks", expanded=False):
            if not hits:
                st.caption("No chunks were retrieved.")
            else:
                for i, (chunk, score) in enumerate(hits, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.caption(f"FAISS squared L2 distance: {score} (lower is closer)")
                    st.text_area(
                        f"chunk_body_{i}",
                        value=chunk,
                        height=160,
                        disabled=True,
                        label_visibility="collapsed",
                    )

        with st.expander("Similarity scores", expanded=False):
            st.caption("FAISS: squared L2 distance (lower is closer). Cross-encoder: higher is more relevant.")
            if hits:
                st.markdown("**FAISS (retrieval order)**")
                st.dataframe(
                    [
                        {"rank": i, "squared_L2_distance": float(s), "preview": (c[:120] + "…") if len(c) > 120 else c}
                        for i, (c, s) in enumerate(hits, start=1)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No FAISS scores (empty retrieval).")
            if reranked:
                st.markdown("**Cross-encoder (after re-ranking)**")
                st.dataframe(
                    [
                        {"rank": i, "cross_encoder_score": float(s), "preview": (c[:120] + "…") if len(c) > 120 else c}
                        for i, (c, s) in enumerate(reranked, start=1)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            elif hits:
                st.caption("No re-ranking scores returned.")

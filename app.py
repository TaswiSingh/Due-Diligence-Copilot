import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Import the existing retrieval engine
from retrieve import (
    load_chunks,
    load_embeddings,
    check_ollama,
    retrieve,
    analyze_question,
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Due Diligence Copilot",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_filing_data():
    chunks = load_chunks()
    embeddings = load_embeddings()

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks ({len(chunks)}) do not match "
            f"embeddings ({len(embeddings)})."
        )

    return chunks, embeddings


# ============================================================
# HEADER
# ============================================================

st.title("📊 Due Diligence Copilot")

st.caption(
    "Evidence-grounded risk analysis from SEC filings"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Document")

st.sidebar.success(
    "Apple 2025 Form 10-K"
)

st.sidebar.markdown(
    """
### Pipeline

1. Question analysis
2. Semantic retrieval
3. Risk classification
4. Evidence extraction
5. Local LLM analysis
6. Citation validation
"""
)

st.sidebar.divider()

st.sidebar.caption(
    "Local model: llama3.2:3b"
)

st.sidebar.caption(
    "Embeddings: all-MiniLM-L6-v2"
)


# ============================================================
# QUESTION
# ============================================================

question = st.text_area(
    "Ask a due-diligence question",
    placeholder=(
        "What are the company's main risk factors, "
        "and which risk does management appear to "
        "consider most significant?"
    ),
    height=120,
)


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "Analyze",
    type="primary",
    use_container_width=False,
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            # ------------------------------------------------
            # Load data
            # ------------------------------------------------

            with st.spinner(
                "Loading filing and embedding model..."
            ):

                chunks, embeddings = load_filing_data()
                embedding_model = load_embedding_model()


            # ------------------------------------------------
            # Check Ollama
            # ------------------------------------------------

            with st.spinner(
                "Checking local LLM..."
            ):

                check_ollama()


            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            with st.spinner(
                "Retrieving relevant filing sections..."
            ):

                selected, topics, intent = retrieve(
                    question=question,
                    chunks=chunks,
                    embeddings=embeddings,
                    query_embedding_model=embedding_model,
                )


            # ------------------------------------------------
            # Analysis
            # ------------------------------------------------

            with st.spinner(
                "Generating evidence-grounded analysis..."
            ):

                answer = analyze_question(
                    question=question,
                    selected=selected,
                    topics=topics,
                    intent=intent,
                )


            # =================================================
            # RESULTS
            # =================================================

            st.success(
                "Analysis complete."
            )

            st.markdown("## Analysis")

            st.markdown(answer)


            # =================================================
            # RETRIEVAL DETAILS
            # =================================================

            with st.expander(
                "View retrieval details"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**Detected topics**"
                    )

                    if topics:
                        st.write(
                            ", ".join(topics)
                        )
                    else:
                        st.write(
                            "general"
                        )

                with col2:

                    st.markdown(
                        "**Detected intent**"
                    )

                    st.write(intent)


                st.markdown(
                    "### Retrieved Evidence"
                )

                for item in selected:

                    chunk = item["chunk"]

                    st.markdown(
                        f"""
**Chunk {chunk.get('chunk_id', 'unknown')}**

- Section: `{chunk.get('section', 'Unknown')}`
- Section chunk: `{chunk.get('section_chunk', 'unknown')}`
- Semantic similarity: `{item['semantic_score']:.4f}`
- Keyword score: `{item['keyword_score']:.4f}`
- Topic score: `{item['topic_score']:.4f}`
- Risk-category score: `{item['risk_category_score']:.4f}`
- Final score: `{item['final_score']:.4f}`
"""
                    )


        except Exception as exc:

            st.error(
                "The analysis could not be completed."
            )

            st.exception(exc)
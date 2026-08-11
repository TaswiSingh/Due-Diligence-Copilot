import ollama


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "llama3.2:3b"


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, retrieved_chunks):

    # --------------------------------------------------------
    # Build SEC filing context
    # --------------------------------------------------------

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Chunk ID: {chunk["chunk_id"]}
Section: {chunk["section"]}

{chunk["text"]}
"""
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Grounded prompt
    # --------------------------------------------------------

    prompt = f"""
You are a financial due-diligence analyst.

You must answer the user's question using ONLY the SEC
filing excerpts provided below.

Do NOT use outside knowledge.
Do NOT invent facts.
Do NOT make assumptions that are not supported by the excerpts.

If the excerpts do not contain enough information, say:

"The provided SEC filing excerpts do not contain enough
information to answer this question."

User question:
{query}

SEC filing excerpts:
{context}

Answer requirements:

1. Identify the most important findings.
2. Explain why each finding matters.
3. Reference the relevant SEC filing section when possible.
4. Keep the answer concise and professional.
5. Separate facts from any uncertainty.
"""

    # --------------------------------------------------------
    # Local Ollama inference
    # --------------------------------------------------------

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_chunks = [
        {
            "chunk_id": 5,
            "section": "Item 1A. Risk Factors",
            "score": 0.8,
            "text": """
            Certain components are obtained from single or
            limited sources. Supply disruptions could affect
            the Company's ability to manufacture products.
            """
        }
    ]

    answer = generate_answer(
        "What is Apple's supply chain risk?",
        test_chunks
    )

    print("\nAnswer:")
    print("=" * 80)
    print(answer)
"""
app.py

Streamlit demo: ask a question, get an answer grounded in Amazon reviews
with linked source reviews.

Run with: streamlit run src/app.py
"""

import streamlit as st

from rag_pipeline import retrieve, build_prompt  # answer_question once LLM is wired up

st.set_page_config(page_title="Amazon Review Q&A", layout="centered")

st.title("Amazon Review Q&A")
st.caption("Ask a question about a product category. Answers are grounded in real customer reviews.")

category = st.text_input("Filter by category (optional)", "")
question = st.text_input("Your question", "Do people complain about battery life?")

if st.button("Ask") and question:
    with st.spinner("Retrieving relevant reviews..."):
        chunks = retrieve(question, category_filter=category or None)

    if not chunks:
        st.warning("No relevant reviews found — try a different question or category.")
    else:
        st.subheader("Retrieved evidence")
        for c in chunks:
            with st.expander(f"{c.chunk_id} — {c.metadata.get('product_name')} (rating: {c.metadata.get('rating')})"):
                st.write(c.text)

        st.subheader("Answer")
        st.info(
            "LLM call not yet wired up — see rag_pipeline.answer_question(). "
            "Once connected, the generated, cited answer will appear here."
        )
        with st.expander("Show assembled prompt (debug)"):
            st.code(build_prompt(question, chunks))

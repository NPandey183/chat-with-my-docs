# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from utils import chunk_text, build_faiss_index, search_faiss

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Chat with My Docs", layout="wide")
st.title("🧠 Chat with My Docs (LLM + RAG)")
st.markdown("Upload a `.txt` file and ask any question based on its content.")

uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
user_question = st.text_input("Ask a question about the document:")

if uploaded_file and user_question:
    file_content = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(file_content)
    index, text_chunks = build_faiss_index(chunks)

    top_chunks = search_faiss(index, user_question, text_chunks)

    context = "\n\n".join(top_chunks)
    prompt = f"Answer based on the following context:\n\n{context}\n\nQuestion: {user_question}"

    with st.spinner("🤖 Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You're a helpful assistant answering based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            answer = response.choices[0].message.content
            st.markdown("### 🧾 Answer")
            st.markdown(answer)

            with st.expander("🧩 Source Chunks Used"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {i}:**\n```\n{chunk}\n```")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

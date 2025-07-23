import streamlit as st
import requests
import json
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

# ---------- Page Configuration ----------
st.set_page_config(page_title="ArticleBot", layout="centered")
st.title("🧠 ArticleBot")

# ---------- Sidebar for LLM Selection ----------
st.sidebar.title("🛠️ Choose a Model")
selected_model = st.sidebar.selectbox(
    "Select a model to generate the article:",
    ["llama3", "mistral", "gemma"],
    index=0
)

# ---------- Setup ChromaDB Vector Store ----------
CHROMA_DIR = "./chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
if "knowledge" not in chroma_client.list_collections():
    collection = collection = chroma_client.get_or_create_collection(name="knowledge")
else:
    collection = chroma_client.get_collection(name="knowledge")

# ---------- Load SentenceTransformer Model ----------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Upload & Index Knowledge ----------
st.sidebar.markdown("### 📄 Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload .txt or .md file", type=["txt", "md"])
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    chunks = content.split("\n\n")  # Simple chunking
    embeddings = embedder.encode(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)
    st.sidebar.success("✅ Document embedded into knowledge base!")

# ---------- Initialize Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Helper: Retrieve Context ----------
def retrieve_context(prompt, top_k=3):
    query_embedding = embedder.encode([prompt])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents)

# ---------- Query Ollama ----------
def query_ollama(prompt, model_name):
    try:
        context = retrieve_context(prompt)
        combined_prompt = f"Use the following context to help answer:\n{context}\n\nUser: {prompt}"

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": combined_prompt}]
            },
            stream=True
        )

        full_reply = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    content = data.get("message", {}).get("content", "")
                    full_reply += content
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
        return full_reply if full_reply else "⚠️ No response from model."

    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {str(e)}"

# ---------- User Input ----------
user_prompt = st.chat_input("Say something...")

# ---------- Display Chat History ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Handle User Prompt ----------
if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {selected_model}..."):
            reply = query_ollama(user_prompt, selected_model)
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

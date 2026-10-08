import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


st.set_page_config(
    page_title="Smart Document Knowledge Assistant",
    page_icon="📚"
)

st.title("📚 Smart Document Knowledge Assistant")
st.write("Upload your documents and ask questions about them.")


# API KEY
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input(
        "Enter Gemini API Key",
        type="password"
    )

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)


# Models
EMBEDDING_MODEL = "gemini-embedding-2"
LLM_MODEL = "gemini-3.7-flash"


# Store data while the app is running
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Create embeddings
def create_embeddings(texts):

    embeddings = []

    for text in texts:

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        embeddings.append(result.embeddings[0].values)

    return np.array(embeddings, dtype=np.float32)


# Find relevant chunks
def search_documents(question, top_k=5):

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question
    )

    question_vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    document_vectors = st.session_state.embeddings

    scores = np.dot(
        document_vectors,
        question_vector
    ) / (
        np.linalg.norm(document_vectors, axis=1)
        * np.linalg.norm(question_vector)
        + 1e-10
    )

    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in best_indices:

        results.append({
            "text": st.session_state.chunks[i]["text"],
            "source": st.session_state.chunks[i]["source"],
            "page": st.session_state.chunks[i]["page"],
            "score": float(scores[i])
        })

    return results


# Process uploaded documents
def process_documents(files):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    all_chunks = []

    for file in files:

        if file.name.lower().endswith(".pdf"):

            reader = PdfReader(file)

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if not text:
                    continue

                pieces = splitter.split_text(text)

                for piece in pieces:

                    all_chunks.append({
                        "text": piece,
                        "source": file.name,
                        "page": page_number
                    })

        elif file.name.lower().endswith(".txt"):

            text = file.read().decode(
                "utf-8",
                errors="ignore"
            )

            pieces = splitter.split_text(text)

            for piece in pieces:

                all_chunks.append({
                    "text": piece,
                    "source": file.name,
                    "page": 1
                })

    if not all_chunks:
        return [], None

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = create_embeddings(texts)

    return all_chunks, embeddings


# SIDEBAR
st.sidebar.header("📄 Documents")

files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


if st.sidebar.button("Process Documents"):

    if not files:

        st.sidebar.warning(
            "Please upload at least one document."
        )

    else:

        with st.spinner(
            "Reading documents and creating embeddings..."
        ):

            chunks, embeddings = process_documents(files)

            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.messages = []

        st.sidebar.success("Documents processed!")

        st.sidebar.write(
            "Chunks created:",
            len(chunks)
        )


# STATUS
if st.session_state.chunks:

    st.success(
        f"Knowledge base ready: "
        f"{len(st.session_state.chunks)} chunks."
    )

else:

    st.info(
        "Upload a PDF or TXT file and click "
        "'Process Documents'."
    )


# SHOW CHAT HISTORY
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# CHAT INPUT
question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    if not st.session_state.chunks:

        st.warning(
            "Please process a document first."
        )

        st.stop()

    # Show question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # Retrieve top 5 chunks
    with st.spinner("Searching documents..."):

        relevant_chunks = search_documents(
            question,
            top_k=5
        )


    # Create context for LLM
    context = ""

    for i, item in enumerate(
        relevant_chunks,
        start=1
    ):

        context += f"""
SOURCE {i}
File: {item["source"]}
Page: {item["page"]}

{item["text"]}

--------------------
"""


    # Prompt
    prompt = f"""
You are a Smart Document Knowledge Assistant.

Answer the user's question using ONLY the
provided document context.

Do not use outside knowledge.

If the answer is not present in the documents,
say:

"I couldn't find that information in the uploaded documents."

Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            try:

                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt
                )

                answer = response.text

                st.markdown(answer)

            except Exception as error:

                answer = "There was an error generating the answer."

                st.error(str(error))


        # Sources
        st.markdown("### 📌 Referenced Sources")

        for i, item in enumerate(
            relevant_chunks,
            start=1
        ):

            with st.expander(
                f"{i}. {item['source']} — Page {item['page']}"
            ):

                st.write(item["text"])

                st.caption(
                    f"Similarity: {item['score']:.3f}"
                )


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
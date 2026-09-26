import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
print("HF TOKEN LOADED:", bool(os.getenv("HF_TOKEN")))
print("HF TOKEN:", os.getenv("HF_TOKEN"))

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")
st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

@st.cache_resource
def get_llm():
    return InferenceClient(
        api_key=os.getenv("HF_TOKEN"),
        provider="auto"
    )

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            embeddings = get_embeddings()

            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

        st.success("Vector database created successfully!")
        st.cache_resource.clear()

if os.path.exists("chroma_db"):
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 8,
            "lambda_mult": 0.5
        }
    )

    llm = get_llm()

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if st.button("Ask Question"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the document..."):
                docs = retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

            prompt = f"""You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{query}

Answer:"""

            try:
                with st.spinner("Generating answer..."):
                    response = llm.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        max_tokens=500
                    )

                answer = response.choices[0].message.content

                st.write("### AI Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"An error occurred: {e}")
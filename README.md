# RAG Book Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF books/documents and ask questions based on their content.

The application retrieves relevant information from the uploaded document using semantic search and MMR-based retrieval, then generates an answer using an LLM.

## Features

- Upload PDF books/documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate semantic embeddings using Hugging Face
- Store document embeddings in ChromaDB
- Retrieve relevant document chunks using MMR
- Generate context-based answers using an LLM
- Prevent unsupported answers when information is not available in the document
- Interactive Streamlit interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Embeddings
- Sentence Transformers
- ChromaDB
- PyPDF
- Hugging Face Inference API
- GPT-OSS-120B

## How It Works

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Hugging Face Embeddings
    ↓
ChromaDB Vector Store
    ↓
MMR Retrieval
    ↓
Relevant Context
    ↓
   LLM
    ↓
AI Generated Answer
```
## RAG Pipeline

### 1. Document Loading

The uploaded PDF is processed using `PyPDFLoader`.

### 2. Text Splitting

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

- Chunk size: 1000
- Chunk overlap: 200

### 3. Embeddings

Each text chunk is converted into a numerical vector using:

`sentence-transformers/all-MiniLM-L6-v2`

### 4. Vector Database

The embeddings are stored in ChromaDB, which allows efficient semantic retrieval.

### 5. Retrieval

The application uses Maximal Marginal Relevance (MMR) retrieval to retrieve relevant and diverse chunks from the document.

### 6. Answer Generation

The retrieved chunks are provided as context to the LLM.

The model is instructed to answer using only the retrieved document context.

If the required information is not available, the application responds:

> I could not find the answer in the document.

## Project Structure

```text
RAG-Book-Assistant/
│
├── app.py
├── create_database.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/arishak10/RAG-Book-Assistant.git
```
### 2. Navigate to the Project
```bash
cd RAG-Book-Assistant
```
### 3. Create a Virtual Environment
```bash
python -m venv .venv
```
### Activate it on Windows:
```bash
.venv\Scripts\activate
```
### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a .env file in the project directory:
```bash
HF_TOKEN=your_huggingface_token
```
Do not upload the .env file to GitHub.

Make sure .env is included in .gitignore.

## Run the Application

Start the Streamlit application using:
```bash
streamlit run app.py
```
The application will open in your browser.

## How to Use
Open the Streamlit application.
Upload a PDF book or document.
Click Create Vector Database.
Wait for the document to be processed.
Enter a question related to the uploaded document.
Click Ask Question.
The application retrieves relevant information and generates an answer.

## Retrieval Configuration

The application uses MMR retrieval with the following configuration:
```bash
search_type="mmr"

search_kwargs={
    "k": 3,
    "fetch_k": 8,
    "lambda_mult": 0.5
}
```
This helps retrieve relevant information while reducing redundant results.

## Context-Based Answering

The LLM is instructed to use only the retrieved document context.

This helps keep responses focused on the uploaded document and reduces unsupported answers.

## Future Improvements

Support for multiple document formats
Chat history and conversational memory
Source and page references in answers
Multiple PDF support
Improved document management
More advanced retrieval strategies
Streaming AI responses
Deployment using Streamlit Cloud

## Author

Arisha Khan

Computer Science Student | AI/ML & Data Science Enthusiast

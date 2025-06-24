# RAG-ChatBot
# Document Question Answering with RAG

This project demonstrates a question answering system that uses Retrieval Augmented Generation (RAG) to answer questions based on user-provided documents (PDF or TXT). It utilizes open-source libraries and models, runnable in environments like Google Colab.

## Features

- **Document Upload:** Easily upload PDF and TXT files for processing.
- **Document Loading and Parsing:** Handles multiple document types.
- **Text Splitting:** Splits documents into manageable chunks for processing.
- **Embedding Creation:** Uses a pre-trained Sentence Transformer model to create document embeddings.
- **Vector Store:** Utilizes FAISS for efficient similarity search.
- **Language Model (LLM):** Employs a small and fast Hugging Face model (Flan-T5-base) for generating answers.
- **Retrieval Augmented Generation (RAG):** Combines document retrieval and LLM generation to provide context-aware answers.
- **Basic QA Chain:** A simple chain for direct question answering.
- **Conversational QA Chain:** A stricter chain that only answers based on provided context and maintains chat history.

## Requirements

- Python 3.6+
- Required Python packages (listed in the code's `!pip install` command)

## Setup and Installation

The easiest way to run this code is in a Google Colab environment.

1. **Open a New Colab Notebook:** Go to [colab.research.google.com](https://colab.research.google.com/).
2. **Copy the Code:** Copy the provided Python code into the Colab notebook cells.
3. **Run the Cells:** Execute the notebook cells sequentially.

Alternatively, you can run this locally:

1. **Clone the Repository:** If the code is in a repository, clone it. Otherwise, save the code as a Python file (`.py`).
2. **Install Dependencies:**
Use code with caution
bash pip install -q faiss-cpu sentence-transformers transformers langchain pypdf accelerate

3. **Run the Script:** Execute the Python file from your terminal.

## Usage

1. **Install Packages and Import Libraries:** Run the initial cells to install the necessary libraries and import modules.
2. **Upload Documents:** When prompted, upload your PDF or TXT files.
3. **Process Documents:** The code will automatically load, split, and embed the documents, creating a searchable vector store.
4. **Ask Questions:**
   - The first QA chain allows you to enter a single question.
   - The Conversational QA chain provides an interactive chat interface. Type your questions and press Enter. To exit the chat, type 'exit'.

The system will retrieve relevant document chunks based on your question and use the LLM to generate an answer. Sources for the answer will also be displayed.

## Code Breakdown

- **Step 1:** Installs required Python packages.
- **Step 2:** Imports necessary libraries.
- **Step 3:** Handles uploading of user files (.pdf, .txt).
- **Step 4:** Loads and parses the uploaded documents.
- **Step 5:** Splits the documents into smaller chunks using a `RecursiveCharacterTextSplitter`.
- **Step 6:** Creates embeddings for the text chunks using `sentence-transformers/all-MiniLM-L6-v2` and builds a FAISS vector store.
- **Step 7:** Loads a pre-trained Hugging Face Language Model (`google/flan-t5-base`) and sets up a text generation pipeline.
- **Step 8:** Creates a basic `RetrievalQA` chain for question answering.
- **Step 9:** Provides an interface to ask a single question using the basic QA chain and formats the output.
- **Step 10:** Sets up a stricter `ConversationalRetrievalChain` with a custom prompt to ensure answers are based only on the provided context. It also maintains chat history and allows for an interactive conversation.

## Customization

- **Chunk Size and Overlap:** Adjust `chunk_size` and `chunk_overlap` in Step 5 to fine-tune how documents are split.
- **Embedding Model:** You can change the `model_name` in Step 6 to use a different Sentence Transformer model.
- **Language Model:** Replace `model_name` in Step 7 to use a different compatible Hugging Face model. Be mindful of model size and computational requirements.
- **Prompt Template:** Modify the `prompt_template` in Step 10 to change the behavior of the conversational QA chain.
- **Retriever Search Parameters:** Adjust `search_kwargs` (e.g., `k`) in Step 10 to control the number of document chunks retrieved.

## Acknowledgements

- [Langchain](https://www.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)


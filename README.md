# 🤖🧠 DocGemini: Retrieval-Augmented QA for Documents 🔗

DocGemini is an intelligent document question-answering chatbot built using Retrieval-Augmented Generation (RAG). It leverages LangChain for document processing and Google Gemini Pro for generating accurate and concise answers based on document content. Simply upload a PDF, and the chatbot will retrieve relevant information and answer your queries based on the content of the document.

## Features
- **Document Upload**: Upload PDF documents directly to the app.
- **RAG-Powered QA**: Uses Retrieval-Augmented Generation (RAG) to answer questions based on document content.
- **Google Gemini Pro Integration**: Employs Google's Gemini Pro for generating high-quality, concise responses.
- **Vector Search**: Pinecone Vector Store is used for document retrieval and semantic search.
- **Streamlit Interface**: Provides an interactive and user-friendly chat interface using Streamlit.

## Tech Stack
- **LangChain**: Handles document processing and chain creation for retrieval-based question answering.
- **Google Gemini Pro**: The underlying language model (LLM) used for generating answers.
- **Pinecone**: Vector store for efficient document retrieval.
- **Streamlit**: For building the web-based chatbot interface.
  
## Getting Started

### Prerequisites
- **Python 3.8+**
- **Pinecone Account**: You'll need API keys from Pinecone.
- **Google API Key**: You'll need an API key for Google Gemini Pro.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/docgemini-rag-qa.git
   cd docgemini-rag-qa

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Set up API Keys:**
   ```bash
   ps.environ["GOOGLE_API_KEY"] = "your_api_key"
   os.environ["PINECONE_API_KEY"] = "your_api_key"

5. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   
## Usage
1. **Upload a PDF:** Use the sidebar to upload a PDF document.
2. **Ask a Question:** Type in your question about the document in the chat input.
3. **Get an Answer:** The chatbot retrieves relevant content from the document and provides a concise, accurate response.
4. **Clear the Chat:** Click the "Clear Chat" button to start a new conversation.

## Demo
![demo](https://github.com/user-attachments/assets/3a798a3d-7591-414d-93c8-2521865a67af)

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the [MIT License](LICENSE).

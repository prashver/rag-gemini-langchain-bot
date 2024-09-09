import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY"

# Page title
st.set_page_config(page_title="RAG Chat App", page_icon="🤖")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Add a new state variable for tracking if the chat should be cleared
if "clear_chat" not in st.session_state:
    st.session_state.clear_chat = False

# App title
st.title("🤖🧠DocGemini: Retrieval-Augmented QA for Documents🔗")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing the PDF..."):
        # Save uploaded file
        with open("temp_pdf_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and process the PDF
        loader = PyPDFLoader("temp_pdf_file.pdf")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        vectorstore = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            index_name="langchain-chatbot"
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=None, timeout=None)

        # Create prompt template
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        st.success("PDF processed successfully! You can now start chatting.")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.clear_chat = True

# Check if clear_chat is True and clear the messages
if st.session_state.clear_chat:
    st.session_state.messages = []
    st.session_state.clear_chat = False  # Reset the clear_chat flag

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask something about the document:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    if st.session_state.rag_chain is not None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({"input": query})
                st.markdown(response["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    else:
        st.error("Please upload a PDF document first.")
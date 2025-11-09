import streamlit as st
import os
import tempfile
import logging
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
import base64


def get_base64_image(image_path): 
    """"Convert local image to base64 for inline HTML display."""
    with open(image_path,"rb") as image_file: 
        return base64.b64encode(image_file.read()).decode()

bhanu_image_base64 =  get_base64_image('bhanu.png')



# ==============================
# CLASS DEFINITION
# ==============================
class MultiFormatRAG:
    def __init__(self, groq_api_key: str):
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }

        os.environ["GROQ_API_KEY"] = groq_api_key

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_documents(self, directory_path: str):
        documents = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension in self.loader_map:
                try:
                    loader = self.loader_map[file_extension](file_path)
                    docs = loader.load()
                    self.logger.info(f"Loaded {file}")
                    documents.extend(docs)
                except Exception as e:
                    self.logger.error(f"Error loading {file}: {e}")
            else:
                self.logger.warning(f"Unsupported format: {file}")
        return documents

    def process_documents(self, documents):
        texts = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def create_qa_chain(self, vectorstore):
        mode = st.radio("Select Mode", ["AI Bhanu (Chat Style)", "Knowledge Assistant"])

        if mode == "AI Bhanu (Chat Style)":
                system_prompt = """
                You are Bhanu Prakash Sen — a highly skilled Flutter Developer and mentor.
                Answer naturally in Bhanu’s tone and style.
                Context:
                {context}
                Question:
                {question}
                """
        else:
                system_prompt = """
                You are an expert assistant. Use only the provided context to answer the question.
                Do not add extra information.
                Context:
                {context}
                Question:
                {question}
                """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_prompt
        )

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )

        return qa_chain

    def query(self, qa_chain, question: str):
        try:
            response = qa_chain.invoke(question)
            return response['result']
        except Exception as e:
            self.logger.error(f"Query error: {e}")
            return f"Error: {e}"

# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="Bhanu AI Clone", page_icon="🤖", layout="wide")

st.title("🤖 Bhanu Prakash Sen — AI Assistant")
st.markdown("Talk to Bhanu’s digital twin. Upload files, ask questions, and get developer-style answers!")

with st.sidebar:
    st.header("📂 Upload Documents")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .docx, .txt, .csv, .html, .md)",
        accept_multiple_files=True
    )
    process_btn = st.button("🔍 Process Documents")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if process_btn:
    if not groq_api_key:
        st.error("Please enter your Groq API key.")
    elif not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Processing documents..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                rag = MultiFormatRAG(groq_api_key)
                docs = rag.load_documents(temp_dir)
                vectore_store = rag.process_documents(docs)
                qa_chain = rag.create_qa_chain(vectore_store)

                st.session_state.qa_chain = qa_chain

        st.success(" Documents processed successfully! You can now chat with Bhanu.")

# Chat Interface
if st.session_state.qa_chain:
    st.divider()
    st.subheader("💬 Chat with Bhanu")

    user_input = st.text_input("Ask a question:")
    if user_input:
        with st.spinner("Thinking..."):
            rag = MultiFormatRAG(groq_api_key)
            answer = rag.query(st.session_state.qa_chain, user_input)

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; margin-top:10px;">
                <img src="data:image/png;base64,{bhanu_image_base64}" width="40" height="40"
                    style="border-radius:50%; margin-right:10px;"/>
                <div style="background-color:#f1f4f9; padding:10px 14px; border-radius:12px; max-width:80%;">
                    <b>Bhanu:</b> {answer}
                </div>
            </div>
            """,
           unsafe_allow_html=True
               )



else:
    st.info("👆 Upload documents and click **Process Documents** to start chatting.")


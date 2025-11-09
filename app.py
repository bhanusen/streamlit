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



# ==============================
# STREAMLIT APP (REVISED SECTION)
# ==============================
st.set_page_config(page_title="Bhanu AI Clone", page_icon="🤖", layout="wide")

st.title("🤖 Bhanu Prakash Sen — AI Assistant")
st.markdown("Talk to Bhanu’s digital twin. Upload files, ask questions, and get developer-style answers!")

# --- Sidebar Setup (Keep as is) ---
with st.sidebar:
    st.header("📂 Upload Documents")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .docx, .txt, .csv, .html, .md)",
        accept_multiple_files=True
    )
    process_btn = st.button("🔍 Process Documents")

# --- Session State Initialization ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # <-- NEW: Initialize chat history

# --- Document Processing Logic (Keep as is) ---
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

# --- Chat Interface (REVISED) ---
if st.session_state.qa_chain:
    st.divider()
    st.subheader("💬 Chat with Bhanu")

    # 1. Display Existing Chat History
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            # Display User Message
            st.markdown(f"""
                <div style="display:flex; justify-content:flex-end; margin-top:10px;">
                    <div style="background-color:#007bff; color:white; padding:10px 14px; border-radius:12px; max-width:80%;">
                        <b>You:</b> {message['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Display AI (Bhanu) Message
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; margin-top:10px;">
                    <img src="data:image/png;base64,{bhanu_image_base64}" width="40" height="40"
                        style="border-radius:50%; margin-right:10px;"/>
                    <div style="background-color:#f1f4f9; padding:10px 14px; border-radius:12px; max-width:80%;">
                        <b>Bhanu:</b> {message['content']}
                    </div>
                </div>
                """,
               unsafe_allow_html=True
            )


    # 2. Handle New User Input
    user_input = st.chat_input("Ask a question...") # Use st.chat_input for better chat feel

    if user_input:
        # Add user message to history immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Rerun to display the user's message instantly
        st.rerun() 

    # 3. Generate and Display AI Response (Only if history was updated by a new user input in this run)
    # The logic below will only run if the app reruns *after* a user input was processed.
    # Since the user input is handled by st.chat_input and causes a rerun, we need to check the last item.
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        
        # To avoid re-querying on every display, we can check if the last item is the user's last message
        # and if we haven't already generated an AI response for it in a previous state.
        # A simpler, robust method is to only generate if the user input triggered the rerun AND the last item is the user's.
        # For simplicity, let's assume the rerun from st.chat_input is the trigger.
        
        last_message = st.session_state.chat_history[-1]
        
        # Check if the last message is the one we just added (user's message) and the AI hasn't responded yet.
        # A better check would be to store a flag, but for now, we rely on the structure.
        # Since we added the user message and reran, if the LLM hasn't run for this turn, we run it now.
        
        # To prevent the AI from answering on *every* render after history is loaded, 
        # you should only trigger the query when a *new* user message is detected that doesn't have a corresponding AI reply.
        
        # For the first pass, let's simplify: We'll only trigger the AI query if the last entry is the user's, 
        # and we'll use a temporary state to stop re-querying.
        
        # ***Self-Correction for Rerun Logic:***
        # The simplest fix to avoid redundant queries on reruns is to only query *after* adding the user message 
        # and then immediately check if an AI response exists for it in the history. If not, generate it.
        
        # Let's modify the structure to handle this:
        
        # **Remove the `if user_input:` block above and replace it with this structure:**
        
        # 1. Display History (as above)
        # 2. Get Input via st.chat_input
        # 3. If input exists:
        #    a. Add user message.
        #    b. Generate AI response.
        #    c. Add AI response.
        #    d. Rerun (or let the reruns naturally handle the display, but query must be inside the check).
        
        # Let's adopt the standard Streamlit chat flow:
        
        # --- REVISED REVISED CHAT INTERFACE ---
        
        # Re-display the history at the top (This is already done above)
        
        # Get Input
        user_input = st.chat_input("Ask a question...")
        
        if user_input:
            # Add user message to history and rerun
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.rerun()
            
        # If the app executed a rerun because of user input, the last item is the user's message.
        # We need to detect if the *last* message is a user message that hasn't been answered yet.
        # Since we are running this block *after* the display loop, if the last item is the user's, we generate the response.
        
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            
            # Extract the question from the latest (user) message
            latest_question = st.session_state.chat_history[-1]["content"]
            
            with st.spinner("Thinking..."):
                # Ensure you use the key stored in session state for the chain
                rag = MultiFormatRAG(groq_api_key) 
                answer = rag.query(st.session_state.qa_chain, latest_question)

            # Add AI response to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            # Rerun again to display the newly added assistant message alongside the user's message
            st.rerun()

else:
    st.info("👆 Upload documents and click **Process Documents** to start chatting.")
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
# REPLACE THE TEXT BELOW WITH YOUR ACTUAL API KEY
# Judges: Please paste your own Google Gemini API Key here
os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_NEW_KEY_HERE"

# --- UI SETUP ---
st.set_page_config(page_title="HR Assistant", layout="wide")
st.title("🤖 HR Policy Assistant")
st.markdown("I answer questions based strictly on our Company Handbook.")

# --- 1. LOAD DATA & CREATE VECTORS ---
@st.cache_resource
def get_vector_store():
    # Load the text file
    loader = TextLoader("hr_policy.txt")
    docs = loader.load()
    
    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create Vector Store (The Brain) using Google Embeddings
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    return vectorstore

# Initialize the Knowledge Base
try:
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"Error loading knowledge base: {e}")
    st.stop()

# --- 2. SETUP THE AI ---
# We use Gemini Flash (Fast & Free)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# The System Prompt restricts the AI to ONLY use the provided context
system_prompt = (
    "You are an HR Assistant. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer is not in the context, say 'I cannot find that in the policy documents.' "
    "Keep answers concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the chain: Retrieval -> Prompt -> LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_input := st.chat_input("Ask about leaves, WFH, or expenses..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response['answer']
            st.markdown(answer)
            
    # Save response
    st.session_state.messages.append({"role": "assistant", "content": answer})
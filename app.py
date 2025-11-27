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
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # Judges: Paste key here if running locally
    os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_NEW_KEY_HERE"

# --- UI SETUP ---
st.set_page_config(page_title="HR Genie", page_icon="🧞", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stButton button {width: 100%; border-radius: 20px;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("⚙️ Controls")
    
    # Feature 1: Response Style (Shows you can control the AI)
    response_style = st.radio(
        "Answer Style:",
        ["Concise & Direct", "Detailed & Explanatory"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**System Status:** 🟢 Online")
    st.markdown("**Knowledge Base:** Employee Handbook v2.1")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN HEADER ---
st.title("🧞 HR Policy Genie")
st.markdown("##### Your 24/7 AI Assistant for Company Policies")

# --- 1. LOAD DATA ---
@st.cache_resource
def get_vector_store():
    loader = TextLoader("hr_policy.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    return vectorstore

try:
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"Error loading knowledge base: {e}")
    st.stop()

# --- 2. SETUP AI ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Dynamic System Prompt based on User Selection
style_instruction = "Keep answers very short and to the point." if response_style == "Concise & Direct" else "Provide detailed explanations with examples where possible."

system_prompt = (
    f"You are an HR Expert. {style_instruction} "
    "Use the following context to answer. "
    "If unknown, say 'I cannot find that in the policy.' "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your HR Genie. Ask me about leaves, remote work, or expenses!"}]

# Display History
for msg in st.session_state.messages:
    avatar = "🧞" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔎 Source Verification"):
                st.info(msg["sources"])

# --- FEATURE 2: QUICK ACTION BUTTONS ---
# This is the "Superb" feature. Clickable questions.
st.markdown("---")
st.write("📌 **Quick Questions:**")
cols = st.columns(4)
buttons = [
    ("🤒 Sick Leave", "What is the policy for sick leave?"),
    ("🏠 WFH Rules", "Can I work from home?"),
    ("✈️ Travel Allowance", "What is the food allowance for travel?"),
    ("👶 Maternity", "Tell me about maternity leave.")
]

user_query = None

# Check if a button is clicked
for i, (label, question) in enumerate(buttons):
    if cols[i].button(label):
        user_query = question

# Check if text input is used
chat_input = st.chat_input("Type your question here...")
if chat_input:
    user_query = chat_input

# Process the query (from button OR text)
if user_query:
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    # 2. Generate Answer
    with st.chat_message("assistant", avatar="🧞"):
        with st.spinner("🧠 Consulting the Handbook..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response['answer']
                
                # Extract Sources
                source_text = ""
                for i, doc in enumerate(response["context"]):
                    source_text += f"**Chunk {i+1}:** {doc.page_content[:200]}...\n\n"
                
                st.markdown(answer)
                with st.expander("🔎 Source Verification"):
                    st.info(source_text)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": source_text
                })
                
                # Force refresh to update chat history immediately
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
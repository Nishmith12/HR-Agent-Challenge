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
    # Judges: Please paste your own Google Gemini API Key here
    os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_NEW_KEY_HERE"

# --- UI SETUP ---
st.set_page_config(page_title="HR Policy Genie", page_icon="🧞", layout="wide")

# --- SIDEBAR (The "Pro" Feel) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("HR Policy Genie")
    st.markdown("---")
    st.markdown("**Status:** 🟢 Online")
    st.markdown("**Model:** Gemini 2.5 Flash")
    st.markdown("**Knowledge Base:** Employee Handbook v2.0")
    
    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN PAGE ---
st.title("🧞 HR Policy Assistant")
st.markdown("Ask me anything about leaves, WFH, or expenses. I show my sources!")

# --- 1. LOAD DATA & CREATE VECTORS ---
@st.cache_resource
def get_vector_store():
    loader = TextLoader("hr_policy.txt")
    docs = loader.load()
    
    # Split text into smaller chunks for better accuracy
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

# --- 2. SETUP THE AI ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

system_prompt = (
    "You are an HR Policy Expert called 'HR Genie'. "
    "Use the retrieved context to answer the question. "
    "If the answer is not in the context, say 'I cannot find that in the policy documents.' "
    "Keep answers professional and concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with custom avatars
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🧞"):
            st.markdown(message["content"])
            # If there are sources stored, show them
            if "sources" in message:
                with st.expander("🔎 View Verified Sources"):
                    st.info(message["sources"])

# Handle new user input
if user_input := st.chat_input("Ex: What is the travel food allowance?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant", avatar="🧞"):
        with st.spinner("Searching Company Policy..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response['answer']
                
                # FEATURE: EXTRACT SOURCES
                # We combine the retrieved text chunks into a single string to show the user
                source_text = ""
                for i, doc in enumerate(response["context"]):
                    source_text += f"**Source Chunk {i+1}:**\n{doc.page_content}\n\n"
                
                st.markdown(answer)
                
                # Show the "Wow" feature
                with st.expander("🔎 View Verified Sources"):
                    st.info(source_text)
                
                # Save response AND sources to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": source_text
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
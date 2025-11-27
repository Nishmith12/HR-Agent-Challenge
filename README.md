# HR Policy Assistant Agent 🤖

## 1. Overview
The HR Policy Assistant is an AI-powered agent designed to instantly answer employee queries regarding company policies, leave entitlements, and remote work guidelines. It utilizes Retrieval-Augmented Generation (RAG) to ensure all answers are strictly based on the official "Employee Handbook," eliminating hallucinations.

## 2. Features & Limitations
* **Accurate Policy Retrieval:** Fetches precise answers from the HR policy document.
* **Context-Aware:** Understands specific HR terminology (e.g., "CL", "WFH").
* **Safe Fallback:** Explicitly states when information is missing rather than guessing.
* **Limitation:** Currently relies on a static text file; does not connect to live HRMS databases yet.

## 3. Tech Stack
* **Frontend:** Streamlit (Python)
* **LLM:** Google Gemini 2.5 Flash
* **Orchestration:** LangChain
* **Vector Database:** FAISS (CPU)
* **Embeddings:** Google Generative AI Embeddings (text-embedding-004)

## 4. Setup & Run Instructions
1.  **Clone the Repository**
    ```bash
    git clone <your-repo-link-here>
    cd HR_Agent_Challenge
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API Key**
    * Open `app.py` and replace `os.environ["GOOGLE_API_KEY"]` with your Gemini API Key.
4.  **Run the Agent**
    ```bash
    streamlit run app.py
    ```

## 5. Architecture
The agent follows a standard RAG pipeline:
User Query -> Vector Search (FAISS) -> Retrieval of Policy Chunks -> Gemini 2.5 Generation -> Answer.
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- Setup (same as before) ---
documents = [
    Document(page_content="John Doe has 5 years of experience in Python and SQL."),
    Document(page_content="Priya Sharma is a React and Node.js developer with 3 years experience."),
    Document(page_content="Arjun Mehta specializes in Java, Spring Boot, and AWS with 7 years experience."),
    Document(page_content="Sara knows HTML, CSS, and basic JavaScript, 1 year experience."),
    Document(page_content="Rahul K is an expert in Python, Django, Docker, and Kubernetes with 10+ years."),
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}""")

# --- 1. Define the State (shared object passed between nodes) ---
class RAGState(TypedDict):
    question: str
    context: str
    answer: str

# --- 2. Define Nodes (each is a function that updates the state) ---
def retrieve_node(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    context = "\n".join(doc.page_content for doc in docs)
    print("[NODE: retrieve] context found:", context)
    return {"context": context}

def generate_node(state: RAGState) -> RAGState:
    chain = prompt | llm
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    print("[NODE: generate] answer produced")
    return {"answer": response.content}

# --- Updated generate_node with a relevance check ---
def check_relevance(state: RAGState) -> str:
    # crude check: if context is empty or very short, treat as "no match"
    if not state["context"].strip():
        return "no_context"
    return "has_context"

def no_context_node(state: RAGState) -> RAGState:
    print("[NODE: no_context] skipping generation")
    return {"answer": "No relevant information found in the knowledge base."}

# --- Rebuild the graph with a conditional edge ---
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("no_context", no_context_node)

graph.set_entry_point("retrieve")

graph.add_conditional_edges(
    "retrieve",
    check_relevance,
    {
        "has_context": "generate",
        "no_context": "no_context"
    }
)

graph.add_edge("generate", END)
graph.add_edge("no_context", END)

app = graph.compile()

print("RAG Chat (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    result = app.invoke({"question": user_input})
    print("Assistant:", result["answer"])
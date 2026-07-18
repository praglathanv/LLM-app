from dotenv import load_dotenv
load_dotenv()

# LangSmith automatically traces any LangChain/LangGraph call
# once LANGCHAIN_TRACING_V2=true is set in .env — no extra code needed

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

documents = [
    Document(page_content="John Doe has 5 years of experience in Python and SQL."),
    Document(page_content="Priya Sharma is a React and Node.js developer with 3 years experience."),
    Document(page_content="Rahul K is an expert in Python, Django, Docker, and Kubernetes with 10+ years."),
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_template("""Answer using ONLY the context below.

Context:
{context}

Question: {question}""")

class RAGState(TypedDict):
    question: str
    context: str
    answer: str

def retrieve_node(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    return {"context": "\n".join(d.page_content for d in docs)}

def generate_node(state: RAGState) -> RAGState:
    chain = prompt | llm
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": response.content}

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()

result = app.invoke({"question": "Who knows Python?"})
print(result["answer"])
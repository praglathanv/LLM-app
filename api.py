from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

documents = [
    Document(page_content="John Doe has 5 years of experience in Python and SQL."),
    Document(page_content="Priya Sharma is a React and Node.js developer with 3 years experience."),
    Document(page_content="Arjun Mehta specializes in Java, Spring Boot, and AWS with 7 years experience."),
    Document(page_content="Sara knows HTML, CSS, and basic JavaScript, 1 year experience."),
    Document(page_content="Rahul K is an expert in Python, Django, Docker, and Kubernetes with 10+ years."),
]

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_template("""Answer using ONLY the context below. If the context doesn't contain the answer, say so.

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
rag_graph = graph.compile()

api = FastAPI(title="Recruiting RAG API")

class Query(BaseModel):
    question: str

@api.get("/")
def root():
    return {"status": "RAG API is running"}

@api.post("/ask")
def ask(query: Query):
    result = rag_graph.invoke({"question": query.question})
    return {"question": query.question, "answer": result["answer"]}
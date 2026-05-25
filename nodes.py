from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from graph_state import GraphState


# When a user asks a question, FAISS needs to convert that question into a vector first before it can compare it against the 42 stored vectors.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "vectorstore/faiss_index",
    embedding_model,
    # FAISS saves the index.pkl file using Python's pickle format. 
    # Pickle is powerful but dangerous — a malicious .pkl file can execute arbitrary code on your machine when loaded.
    allow_dangerous_deserialization=True
)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs = {"k":4}
)

def retrieve(state:GraphState):
    print("Retrieve node")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents":documents,"question":question}



import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, yes or no")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score: yes or no."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: {document} | User question: {question}"),
])

# This is LangChain's chain syntax. It connects components so output of one becomes input of the next — exactly like a pipe in a factory assembly line.
retrieval_grader = grade_prompt | structured_llm_grader

def grade_documents(state: GraphState):
    print("--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    retries = state.get("retries", 0)

    filtered_docs = []

    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        if score.binary_score == "yes":
            filtered_docs.append(doc)

    if not filtered_docs:
        web_search = "Yes"
        print("No relevant documents found, will rewrite query")
    else:
        web_search = "No"
        print(f"Found {len(filtered_docs)} relevant documents")

    return {"documents": filtered_docs, "question": question, "web_search": web_search, "retries": retries}



## generation node



from langchain_core.output_parsers import StrOutputParser

system = """You are an assistant for question-answering tasks.
Use ONLY the following retrieved context to answer the question.
If the context does not contain information to answer the question, 
respond with exactly: I don't have information about this in the provided documents.
Do not make up answers. Keep the answer concise with a maximum of 3 sentences."""


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Context: {context} | Question: {question}"),
])
## Takes the list of chunk objects and joins their text content into one single string separated by double newlines.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = rag_prompt | llm | StrOutputParser()

def generate(state: GraphState):
    print("--- NODE: GENERATE ---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({
        "context": format_docs(documents),
        "question": question
    })
    return {"documents": documents, "question": question, "generation": generation}



## Hallucination and Answer Grading


class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in facts, yes or no")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, yes or no")

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader checking if an LLM answer is grounded in retrieved facts.
Give a binary score: yes if the answer is grounded in the documents, no if it contains made up facts."""),
    ("human", "Retrieved facts: {documents} | LLM answer: {generation}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader checking if an answer resolves the user question.
Give a binary score: yes if the answer resolves the question, no if it does not."""),
    ("human", "User question: {question} | LLM answer: {generation}"),
])

hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)
answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)



## Query Rewritting

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a question rewriter that improves questions for vector store retrieval.
Look at the input question and reason about the underlying semantic intent.
Rewrite it to be more specific and better suited for document retrieval."""),
    ("human", "Original question: {question} | Rewrite it:"),
])

question_rewriter = rewrite_prompt | llm | StrOutputParser()

def transform_query(state: GraphState):
    print("--- NODE: TRANSFORM QUERY ---")
    question = state["question"]
    retries = state.get("retries",0)+1
    print("Retry attempt.......!")
    better_question = question_rewriter.invoke({"question": question})
    print(f"Original question: {question}")
    print(f"Rewritten question: {better_question}")
    return {"documents": state["documents"], "question": better_question,"retries":retries}



import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_and_split():
    loader = Docx2txtLoader("data/rag_guide.docx")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = splitter.split_documents(documents=documents)


    # print(f"total documents {len(documents)}")
    # print(f"total chunks {len(chunks)}")
    # print(f"first content {chunks[0].page_content}")
    # print(f"metadata {chunks[0].metadata}")

    return chunks


def create_vectorstore(chunks):
    print("\n Loading embedding model....")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    print("Creating vector store")

    vectorstore = FAISS.from_documents(chunks,embedding_model)

    os.makedirs("vectorstore",exist_ok=True)
    vectorstore.save_local("vectorstore/faiss_index")
    print("vectorstore saved ")

    return vectorstore

if __name__ =="__main__":
    chunks = load_and_split()
    create_vectorstore(chunks=chunks)




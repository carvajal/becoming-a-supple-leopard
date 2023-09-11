from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file to load OPENAI_API_KEY

from langchain.document_loaders import PyPDFLoader
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

if __name__ == "__main__":
    loader = PyPDFLoader("documents/becoming-a-supple-leopard.pdf")
    pages = loader.load_and_split()

    persist_directory = 'becoming-a-supple-leopard-chromadb/'
    embedding = OpenAIEmbeddings()

    client = chromadb.PersistentClient(path=persist_directory)
    vectordb = Chroma.from_documents(
        documents=pages,
        embedding=embedding,
        client=client
    )
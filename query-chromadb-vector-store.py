from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file to load OPENAI_API_KEY

import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

chromadb_path = 'becoming-a-supple-leopard-chromadb/'

# Load the persisted embeddings vectors store from a local DB. 
client = chromadb.PersistentClient(path=chromadb_path)
vectordb = Chroma(
    client=client,
    embedding_function=OpenAIEmbeddings()
)

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
retrieval_qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever(), return_source_documents=True)

from pprint import pprint

def print_response_and_sources(response):
    print(response['result'])
    if not "I don't know" in response['result']:
        print('\n\nSources:')
        for source in response["source_documents"]:
            pprint(f"{source.metadata['source']} - page: {source.metadata['page']}")

if __name__ == "__main__":
    print("Hi. I'm a RetrievalQA bot that answers questions based on the contents of Becoming a Supple Leopard. I'm stateless so make each question as specific as possible. Anything you want to ask?")
    while True:
        question = input()
        if question.lower() in ['goodbye', 'bye', 'exit', 'quit', 'no', 'nah']:
            print("Goodbye!")
            break

        response = retrieval_qa(question)
        print_response_and_sources(response)
        print('Anything else you want to ask?')

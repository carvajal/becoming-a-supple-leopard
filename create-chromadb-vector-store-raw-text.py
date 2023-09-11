from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file to load OPENAI_API_KEY

from PyPDF2 import PdfReader
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

if __name__ == "__main__":
    doc_reader = PdfReader('documents/becoming-a-supple-leopard.pdf')

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # remove redundant whitespaces without loosing lines
    lines = raw_text.split('\n')
    for i in range(0, len(lines)):
        lines[i] = ' '.join(lines[i].split())
    raw_text = '\n'.join(lines)

    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = 'becoming-a-supple-leopard-chromadb/'

    ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
    embedding = OpenAIEmbeddings()

    client = chromadb.PersistentClient(path=persist_directory)
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        client=client
    )
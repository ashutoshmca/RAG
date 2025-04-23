from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv

load_dotenv()


pdf_path = Path(__file__).parent /<your_pdf_file_name>.pdf
#For Jupiter use: pdf_path = Path().resolve() / "ArchitectureAssessmentFrameworks.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = QdrantVectorStore.from_documents(
     documents=[],
     url="http://localhost:6333",
    collection_name="learning_langchain",
     embedding=embedder
)

vector_store.add_documents(documents=split_docs)
print("Indexing Done")
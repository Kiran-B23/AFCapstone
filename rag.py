from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = "chroma_store"

def build_library(docs_dir="data/docs"):
    docs = DirectoryLoader(docs_dir, glob="**/*.*", loader_cls=TextLoader).load()
    if not docs:
        raise SystemExit(f"No documents found in {docs_dir!r} - add some .txt/.md files and re-run.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    Chroma.from_documents(chunks, EMBEDDINGS, persist_directory=CHROMA_DIR)  # saves to disk
    print(f"Indexed {len(chunks)} chunks into ChromaDB")

def get_retriever(k=4):
    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)
    return store.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    build_library()   # run once: python rag.py
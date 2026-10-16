import os
import streamlit as st
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1 : Page Configuration
st.set_page_config(page_title="C++ RAG ChatBot", page_icon="🤯")
st.title("🤯 C++ RAG ChatBot")
st.write("Ask any questions relaed to C++ introduction")

# Step 2: Load env variables
load_dotenv()

# Step 3 : Catch document loading
@st.cache_resource
def load_vector_store():
    try :
        # Step a : Load documents
        loader = TextLoader(
        "cpp_intro.txt",
        encoding="utf-8"
        )
        documents = loader.load()

        # Step b : Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 200,
            chunk_overlap = 20
            # Chunk Overlap = 20 characters overlap
            # Overlap helps maintain context continuity
        )
        
        final_documents = text_splitter.split_documents(documents)
        
        # Step c : Embeddings
        embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Step d : Create FAISS Vector store
        # Converts each chunk to embedding, then stores them and makes searchable
        db = FAISS.from_documents(final_documents, embedding)
        # return faiss database
        return db
      
    except Exception as e:
        st.error(f"Real error: {e}")  
        

# Vector database runs only once because of cache concepts
db = load_vector_store()

# User input
user_input = st.text_input("Enter your question about cpp")

if(user_input) :
    # Converts user quesyions to embeddings
    # Searches FAISS database
    # Returns top 3 similar chunks
    document = db.similarity_search(user_input, k=3)
    
    st.subheader("📓 retrieved context")
    
    for i, doc in enumerate(document) :
        st.markdown(f"**Result {i+1} : **")
        st.write(doc.page_content)
        
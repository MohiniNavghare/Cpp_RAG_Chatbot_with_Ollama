# import os
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama 
# from langchain_core.output_parsers import StrOutputParser

# st.set_page_config(page_title="C++ RAG Chatbot", layout="wide")
# st.title("C++ RAG Chatbot")

# load_dotenv()

# @st.cache_resource
# def  load_vector_source() :
#     loader = TextLoader("cpp_intro.txt", encoding="utf-8")
#     document = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 200,
#         chunk_overlap = 20
#     )
    
#     final_document = text_splitter.split_documents(document)
    
#     embedding = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
    
#     db = FAISS.from_documents(final_document, embedding)

#     return db

# db = load_vector_source()
# llm = Ollama(model_name = "gemma2:2b")

# # Chat Interface
# text_input = st.text_input("Ask question about cpp")
# if(text_input) :
#     with st.spinner(text="Thinkin...") :
#         docs = db.similarity_search(text_input)
#         context = "\n".join([doc.page_content for doc in docs])

#     prompt = """Anser the question using the context below
#     Context : {context}
#     Question : {text_input}
#     Answers : 
# """
# response = llm.invoke({prompt})
    
# st.subheader()
# st.write("Answer")
    
    
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="C++ RAG Chatbot", layout="wide")
st.title("C++ RAG Chatbot")

load_dotenv()

@st.cache_resource
def load_vector_source():
    loader = TextLoader("cpp_intro.txt", encoding="utf-8")
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_document = text_splitter.split_documents(document)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(final_document, embedding)
    return db


db = load_vector_source()

llm = Ollama(model="gemma2:2b")   # confirm using: ollama list

text_input = st.text_input("Ask question about C++")

if text_input:
    with st.spinner("Thinking..."):
        docs = db.similarity_search(text_input, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {text_input}

        Answer:
        """

        response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response)
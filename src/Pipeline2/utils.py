from PyPDF2 import PdfReader

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
load_dotenv()

import re
import en_core_web_sm
from transformers import pipeline
        
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()

    cleaned_text=re.sub(r'\s+', ' ',text)
    cleaned_text = ' '.join(text.split())
    return cleaned_text

# !python3 -m spacy download en_core_web_sm
# !pip3 install -U spacy
def text_segment(clean_text):
    nlp = en_core_web_sm.load()
    # nlp=spacy.load('en_core_web_sm')
    doc=nlp(clean_text)
    lst=[]
    for sent in doc.sents:
        lst.append(sent.text)
    return lst


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def intent_classifier(document):
  label=[]
  for sent in document:
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")
    sequence_to_classify = sent
    candidate_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Suprised","excited"]
    op = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    label.append(op['labels'][0])
  return label
    
    

def get_vectorstore(chunks):
    # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    persist_directory = 'db'
    collection_name = 'pdf'
    vectordb = Chroma.from_texts(texts=chunks, 
                                 embedding=embeddings,
                                 persist_directory=persist_directory,
                                 collection_name=collection_name)
    return vectordb


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

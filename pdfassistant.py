import streamlit as st
from PyPDF2 import PdfReader
import langchain
import docx
import pandas as pd
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import GooglePalm, OpenAI
from langchain.embeddings import GooglePalmEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain.chains.question_answering import load_qa_chain

import os

from PIL import Image



api_key1 = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = api_key1


def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    raw_text = ' '.join(allText)
    return raw_text

    
def get_csv_text(file):
    return "Empty"


st.set_page_config(page_title="LIAR, EAT HARAM", page_icon="books")
st.title("Julien sik")



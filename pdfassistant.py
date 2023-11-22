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


st.set_page_config(page_title="Julien sik", page_icon="books")
st.title("Julien sik")



#llm = ChatGooglePalm(temperature=temperature, max_output_tokens= 512 ,verbose=True,streaming=True)
llm = ChatOpenAI(temperature=0.9, model= "gpt-3.5-turbo-1106", verbose=True)

folder_path = "./files"

# Initialize an empty list to store the file paths
file_paths = []

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    # Create the full file path by joining the folder path and the filename
    file_path = os.path.join(folder_path, filename)
    # Append the file path to the list
    file_paths.append(file_path)


@st.cache_resource(show_spinner=False)
def processing_pdf_docx_files(file_paths):
    with st.spinner(text="Getting Ready"):
        # Read text from the provided PDF and DOCX files

        raw_text = '->\n'
        for file_path in file_paths:
            split_tup = os.path.splitext(file_path)
            file_extension = split_tup[1]
            if file_extension == ".pdf":
                pdfreader = PdfReader(file_path)
                raw_text += ''.join(page.extract_text() for page in pdfreader.pages if page.extract_text())
            elif file_extension == ".docx":
                raw_text += get_docx_text(file_path)  # Use your DOCX processing function here
            else:
                raw_text += get_csv_text(file_path)  # Use your CSV processing function here

        # Split the text using Character Text Splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create a FAISS index from texts and embeddings
        document_search = FAISS.from_texts(texts, embeddings)

        return document_search




if 'history' not in st.session_state:  
        st.session_state['history'] = []


if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state["messages"]= []
    st.session_state['history']  = []


########--Main PDF--########



if file_paths is not None:
    
    
    document_search = processing_pdf_docx_files(file_paths)
    
    for msg in st.session_state.messages:
        if msg["role"] == "Assistant":
            st.chat_message("assistant", avatar="🧑‍💻").write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])

    
    if prompt := st.chat_input(placeholder="Ask Me!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
    
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "", ai_prefix= "")

        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                current_message = st.session_state.messages[i]
                next_message = st.session_state.messages[i + 1]
                
                current_role = current_message["role"]
                current_content = current_message["content"]
                
                next_role = next_message["role"]
                next_content = next_message["content"]
                
                # Concatenate role and content for context and output
                context = f"{current_role}: {current_content}\n{next_role}: {next_content}"
                
                memory.save_context({"question": context}, {"output": ""})


            
        prompt_template = r"""
-You are a helpful teacher who can speak chinese.
- Help out the student.
-talk humbly. Answer the question from the provided course context.
-Use the following pieces of context to answer the question at the end.
-this is the context:
---------
{context}
---------

This is chat history between you and user: 
---------
{chat_history}
---------

New Question: {question}

Answer: 
"""


        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "chat_history"]
        )

        # Run the question-answering chain
        docs = document_search.similarity_search(prompt, k=8)

            # Load question-answering chain
        chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
            
        #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")

        with st.chat_message("assistant", avatar="🧑‍💻"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
            response = chain.run(input_documents=docs, question = prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "Assistant", "content": response})
            
            st.write(response)





hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    


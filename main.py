import streamlit as sl
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import os
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain # for chatting
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    
    text = ''
    # for pdf in pdf_docs:
    #     # try:
    #         with pdfplumber.open(pdf) as pdf_reader:
    #             for page in pdf_reader.pages:
    #                 text += page.extract_text()
    #     # except Exception as e:
    #     #     print(f"Error reading {pdf}: {e}. Skipping this file.")

    # for pdf in pdf_docs:
    #     with open(pdf, 'rb') as f:
    #         pdf_data = f.read()
    #     pdf_reader = PdfReader.fromBytes(pdf_data)
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()
    
    # for pdf in pdf_docs:
        
    #     pdf_reader=PdfReader(io.BytesIO(pdf))   
    #     #Create a list to read every page
    #     for page in pdf_reader.pages:
    #         text+=page.extract_text()

    # for pdf in pdf_docs:
    #     pdf_reader = PdfReader(io.BytesIO(pdf))
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()

    # for pdf in pdf_docs:
    #     pdf_document = fitz.open(stream=pdf, filetype="pdf")
    #     for page_num in range(pdf_document.page_count):
    #         page = pdf_document.load_page(page_num)
    #         text += page.get_text()

    # for pdf in pdf_docs:
    #     if isinstance(pdf, str):
    #         try:
    #             pdf_reader = PdfReader(pdf)
    #             # Ensure the PDF has pages
    #             if pdf_reader.pages:
    #                 for page in pdf_reader.pages:
    #                     page_text = page.extract_text()
    #                     if page_text:  # Only add non-empty text
    #                         text += page_text
    #         except Exception as e:
    #             print(f"Error reading {pdf}: {e}. Skipping this file.")
    #     else:
    #         print("Unknown document type. Skipping this file.")

    for pdf in pdf_docs:
        if isinstance(pdf, bytes):
            pdf_document = fitz.open(stream=pdf, filetype="pdf")
            if pdf_document.page_count > 0:  # Check if PDF contains pages
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()
                    if page_text:
                        text += page_text
        else:
            print(f"Skipping unknown document type: {pdf}")
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) #Token containing 10000 words
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectors(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_conversation_chain():
    prompt_temp=""" 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the
    provided context just say, "Answer is not available in the context/ document", Don't provide wrong answers. \n\n
    Context: \n{context}?\n
    Question: \n{question}\n

    Answer: 
    """
    model=ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt=PromptTemplate(template=prompt_temp, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff", prompt=prompt)  #"Stuff" helps in summarization
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db=FAISS.load_local("faiss_index",embeddings)
    docs=new_db.similarity_search(user_question)
    chain=get_conversation_chain()
    response=chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    print(response)
    sl.write("Reply: ", response["output_text"])


def main():
    sl.set_page_config("Chat with your PDFs")
    sl.header("Chat with PDFs using GEMINI 💬")

    user_question=sl.text_input("Ask any question from your pdf file/ files:")

    if user_question:
        user_input(user_question)
    
    with sl.sidebar:
        sl.title("Menu:")
        pdf_docs=sl.file_uploader("Upload pdf files and click on submit button")
        if sl.button('Submit & Process'):
            with sl.spinner("Generating....."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vectors(text_chunks)
                sl.success("Completed!")



if __name__=="__main__":
    main()

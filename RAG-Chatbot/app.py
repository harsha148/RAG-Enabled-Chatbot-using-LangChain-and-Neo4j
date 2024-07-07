import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def get_pdf_content(pdf_docs):
    content = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content += page.extract_text()
    return content

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_embeddings')

def conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    model = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    return load_qa_chain(model,"stuff",prompt=prompt)

def generate_response(user_question):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.load_local('faiss_embeddings',embeddings,allow_dangerous_deserialization=True)
    chain = conversational_chain()
    related_docs = vector_db.similarity_search(user_question)
    response = chain({"input_documents":related_docs,"question":user_question})
    return response

def streamlit_ui():
    st.set_page_config('Chat PDF')
    st.header('Chat with your PDF using ChatGPT')
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        response = generate_response(user_question)
        st.write("Reply: ", response["output_text"])
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_content(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    streamlit_ui()


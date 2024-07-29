#%%writefile app.py

import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import os

st.title("LLM Chatbot News Insights")
st.sidebar.title("News Articles URLs")

# st.set_page_config(page_title = 'News Research Tool')
st.markdown("""
## Get instant insights from online news
## Follow the instructions

1. Get your Google API key from here - https://ai.google.dev/gemini-api/docs/api-key
2. Copy Paste the online news links. 
3. Ask your question.
4. Press "Process URLs" button.
""")

api_key = st.text_input('Enter your Google API key: ', type='password', key='api_key_input')

def get_text_from_urls(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        return data
    except Exception as e:
        st.error(f"Error loading URLs: {e}")
        return None

def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=['\n\n', "\n", '.', ' ']
    )
    docs = text_splitter.split_documents(data)
    return docs

def get_vector_store(docs, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                  google_api_key=api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        os.makedirs('faiss_index', exist_ok=True)
        
        vectorstore.save_local('faiss_index')
        # st.success("Vector store saved successfully.")
        # vectorstore.save_local('faiss_index')
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain(api_key):
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details,
        if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        if not os.path.exists('faiss_index'):
            st.error("Vector store index file not found. Please process URLs first.")
            return

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response['output_text'])
        return response['output_text']
    except Exception as e:
        st.error(f"Error processing user input: {e}")

def main():
    st.header("LLM Chatbot News Insights")
    user_question = st.text_input("Ask your question: ", key="user_question")
    if user_question and api_key:
        user_input(user_question, api_key)

    urls = []
    for i in range(1):
        url = st.sidebar.text_input(f"URL {i+1}")
        if url:
            urls.append(url)

    main_placeholder = st.empty()
    process_url_clicked = st.sidebar.button("Process URLs")
    if process_url_clicked and api_key and urls:
        with st.spinner("Processing URLs..."):
            data = get_text_from_urls(urls)
            if data:
                # st.write("Loaded data:", data)  # Add this line to inspect the data structure
                docs = get_text_chunks(data)
                if docs:
                    get_vector_store(docs, api_key)
                    st.success("Done")

if __name__ == "__main__":
    main()


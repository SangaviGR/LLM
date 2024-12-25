import os
import fitz  # PyMuPDF for PDF parsing
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import google.generativeai as genai
# Initialize Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="Document-Based Chatbot", page_icon=":robot:", layout="wide")

# Initialize Generative Model
model = genai.GenerativeModel('gemini-pro')

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        # Open the PDF using fitz
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"


# Function to generate synonyms or alternative phrasings
def generate_synonyms(user_prompt, previous_query=None):
    try:
        # Construct the prompt for synonym generation
        synonym_template = f"Generate synonyms or alternative phrasings for the following user prompt: '{user_prompt}'"
        if previous_query:
            synonym_template += f" Considering the previous question: '{previous_query}'"
        
        # Generate synonyms/alternative phrasings from the model
        synonym_response = model.generate_content(synonym_template)
        synonyms = synonym_response.text.strip().split("\n")
        
        # Remove duplicates and clean the list
        synonyms = list(set(synonyms))
        
        return synonyms
    except Exception as e:
        st.error(f"An error occurred while generating synonyms: {e}")
        return []


# Function to generate response based on the document and query
def generate_response_from_pdf(user_prompt, document_text, synonyms):
    try:
        # Include synonyms in the prompt
        synonyms_text = ", ".join(synonyms) if synonyms else "No synonyms generated"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(document_text)

        # Create Chroma vector store
        vector_store = Chroma.from_texts(text_chunks, embeddings)

        # Retrieve relevant documents
        retriever = vector_store.as_retriever()
        docs = retriever.get_relevant_documents(user_prompt)

        # Prepare QA Chain
        context_from_docs = "\n".join([doc.page_content for doc in docs])
        qa_prompt = PromptTemplate(
            template="{context}\n\n{question}",
            input_variables=["context", "question"],
        )
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

        # Generate the response
        result = qa_chain.run(context=context_from_docs, question=user_prompt)

        # Validate the response
        if not any(phrase in document_text for phrase in result.split()):
            return "The exact answer isn't available in the document. Refer to the content for more details."

        return result
    except Exception as e:
        return f"Error generating response: {e}"


# Streamlit interface
def main():
    st.title("Chatbot for Document Interaction")

    # Upload a PDF file
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from the PDF
        pdf_text = extract_pdf_text(uploaded_file)

        # Check for errors in extraction
        if "Error" in pdf_text:
            st.error(pdf_text)
        else:
            # Display extracted text preview
            st.text_area("Extracted Text (Preview)", pdf_text[:1000], height=200)

            # Get user query
            user_prompt = st.text_input("Ask a question related to the document:")

            if user_prompt:
                with st.spinner("Processing..."):
                    # Generate synonyms
                    synonyms = generate_synonyms(user_prompt)

                    # Display generated synonyms
                    st.markdown("### Generated Synonyms / Alternative Phrasings:")
                    st.write("\n".join(synonyms))

                    # Generate response based on query and document
                    response = generate_response_from_pdf(user_prompt, pdf_text, synonyms)

                    # Display response
                    st.markdown("### Response:")
                    st.write(response)


# Run the application
if __name__ == "__main__":
    main()

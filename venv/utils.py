from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.prompts import PromptTemplate

from langchain.chains.question_answering import load_qa_chain 
import os

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI


# Configure Google Generative AI
genai.configure(api_key="AIzaSyARvvaaT1CRma9v-I6168xp17e2rFCYB0I")
os.environ["GOOGLE_API_KEY"] = "AIzaSyARvvaaT1CRma9v-I6168xp17e2rFCYB0I"


def get_model_response(file, query):
    try:
        # Split the context text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in file)
        data = text_splitter.split_text(context)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        searcher = Chroma.from_texts(data, embeddings).as_retriever()
        records = searcher.get_relevant_documents(query)
        
        print(records)  # Check if records are being retrieved properly

        # Define the prompt template
        prompt_template = """
        You have to answer the question from the provided context and make sure that you provide all the details.

        Context: {context}

        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        # Prepare input for the chain
        input_data = {
            "input_documents": records,
            "question": query
        }
        
        response = chain(input_data, return_only_outputs=True)
        output_text = response.get('output_text', 'No response generated.')
        return output_text
    
    except Exception as e:
        print(f"Error in get_model_response: {e}")
        return "An error occurred while processing the request."

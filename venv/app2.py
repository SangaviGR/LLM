import os
import fitz  # PyMuPDF for PDF parsing
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Configure Google Generative AI (Gemini)
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Generative Model
model = genai.GenerativeModel('gemini-pro')

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        # Open the uploaded file directly from Streamlit's in-memory file
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Function to generate synonyms or alternative phrasings for the user query
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

# Function to generate response using Gemini based on the PDF content and synonyms
def generate_response_from_pdf(user_prompt, document_text, synonyms):
    try:
        # Include synonyms in the prompt
        synonyms_text = ", ".join(synonyms) if synonyms else "No synonyms generated"
        full_prompt = (
            f"Document: {document_text}\n\n"
            f"User Query: {user_prompt}\n"
            f"Synonyms or alternative phrasings for the query: {synonyms_text}\n\n"
            f"Instructions: Provide an answer that directly references the content of the document. "
            f"If the answer is not explicitly found in the document, state that the information is unavailable in the source material. "
            f"Ensure all responses are factually grounded in the document and avoid generating unverifiable or unrelated information."
        )
        
        # Generate the response
        response = model.generate_content(full_prompt)
        response_text = response.text.strip()
        
        # Verify that the response references the source document
        if not any(phrase in document_text for phrase in response_text.split()):
            return (
                "The exact answer isn't available in the document. "
                "Please refer to the document's content for verification."
            )
        
        return response_text
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit interface to upload the PDF and interact with the chatbot
def main():
    st.set_page_config(page_title="Document-Based Chatbot", page_icon=":robot:", layout="wide")

    st.title("Chatbot for Document Interaction")

    # Upload a PDF file
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from PDF
        pdf_text = extract_pdf_text(uploaded_file)

        # Display the extracted text (first 1000 characters for brevity)
        st.text_area("Extracted Text (Preview)", pdf_text[:1000], height=200)

        # Text area for user query
        user_prompt = st.text_input("Ask a question related to the document:")

        if user_prompt:
            with st.spinner("Processing..."):
                # Generate synonyms or alternative phrasings for the user prompt
                synonyms = generate_synonyms(user_prompt)

                # Display the generated synonyms
                st.markdown("### Generated Synonyms / Alternative Phrasings:")
                st.write("\n".join(synonyms))

                # Generate a response based on the user query and the document content
                response = generate_response_from_pdf(user_prompt, pdf_text, synonyms)

                # Display the response
                st.markdown("### Response:")
                st.write(response)

if __name__ == "__main__":
    main()

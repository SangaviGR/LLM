import streamlit as st
from langchain_community.document_loaders import CSVLoader
import tempfile
import pandas as pd
from utils import get_model_response

def main():
    st.title("Chat with CSV using Gemini Pro")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    # Checking if a file has been uploaded
    if uploaded_file is not None:
        # Use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Debugging: Show the first few rows using Pandas
        try:
            # Check the file contents using Pandas to ensure it's properly formatted
            uploaded_file.seek(0)  # Reset file pointer to the start
            df = pd.read_csv(uploaded_file)
            st.write("CSV File Preview:")
            st.write(df.head())  # Display the first few rows of the CSV

            # Manually load CSV with Pandas before passing to CSVLoader
            uploaded_file.seek(0)  # Reset file pointer to the start
            try:
                # Try loading the file manually using Pandas to confirm the format
                manual_data = pd.read_csv(uploaded_file)
                st.write("Pandas loaded data preview:")
                st.write(manual_data.head())
                st.write(f"Total rows in Pandas Data: {len(manual_data)}")
            except Exception as e:
                st.error(f"Failed to load CSV using Pandas: {e}")

            # Initialize CSVLoader
            csv_loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})

            # Load data into csv Loader
            data = csv_loader.load()

            # Debugging: Check the loaded data to see its structure
            st.write("Raw data loaded by CSVLoader:")
            st.write(data)  # Display raw data loaded by CSVLoader

            # Check if data is empty or not
            if data:
                st.write(f"Total records loaded: {len(data)}")
                # Initialize chat Interface
                user_input = st.text_input("Your Message:")

                # Only call the model if there is valid user input
                if user_input:
                    response = get_model_response(data, user_input)
                    st.write(response)
            else:
                st.error("No data found in the CSV file.")

        except Exception as e:
            st.error(f"An error occurred while loading the CSV: {e}")

if __name__ == "__main__":
    main()

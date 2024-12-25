import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
import pandas as pd
import sqlite3

# Function to execute the SQL query on SQLite database
def execute_sql_query(sql_query, df, table_name):
    try:
        # Create an in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        # Clean columns for SQL compatibility
        cleaned_columns = [col.replace(" ", "_").replace("(", "_").replace(")", "_") for col in df.columns]
        columns_str = ', '.join(cleaned_columns)

        # Create the table dynamically with cleaned column names
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});"
        cursor.execute(create_table_query)

        # Insert data from DataFrame into the table
        for _, row in df.iterrows():
            values = tuple(row)
            insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({', '.join(['?'] * len(values))});"
            cursor.execute(insert_query, values)

        # Execute the generated SQL query
        cursor.execute(sql_query)
        results = cursor.fetchall()

        # Commit changes and close connection
        conn.commit()

        # Convert results into a pandas DataFrame for easy display
        result_columns = [description[0] for description in cursor.description]
        result_df = pd.DataFrame(results, columns=result_columns)

        conn.close()
        return result_df
    except Exception as e:
        return f"Error occurred while executing the SQL query: {e}"

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Generative Model
model = genai.GenerativeModel('gemini-pro')

def generate_synonyms(user_prompt, previous_query=None):
    try:
        # Include the previous query in the prompt if available
        synonym_template = f"Generate synonyms or alternative phrasings for the following user prompt: '{user_prompt}'"
        if previous_query:
            synonym_template += f" Considering the previous question: '{previous_query}'"
        synonym_response = model.generate_content(synonym_template)
        synonyms = synonym_response.text.strip().split("\n")
        return synonyms
    except Exception as e:
        st.error(f"An error occurred while generating synonyms: {e}")
        return []

def generate_sql_query(user_prompt, synonyms, table_name, columns, previous_query=None):
    try:
        # Clean up column names to make them alphanumeric
        cleaned_columns = [col.replace(" ", "_").replace("(", "_").replace(")", "_") for col in columns]

        # Combine user prompt, synonyms, and previous query for query generation
        combined_prompts = f"User prompt: {user_prompt}\n"
        if previous_query:
            combined_prompts += f"Previous query: {previous_query}\n"
        combined_prompts += "Synonyms or alternative phrasings:\n"
        for synonym in synonyms:
            combined_prompts += f"- {synonym}\n"

        template = f"""
        Generate a valid SQL query based on the following table '{table_name}' and its columns:
        {', '.join(cleaned_columns)}
        The query should accurately satisfy this combined request:
        {combined_prompts}
        Ensure the SQL query is executable and specific to the provided table and columns.
        """

        response = model.generate_content(template)
        sql_query = response.text.strip()

        # Clean the SQL query output
        if sql_query.startswith('```sql'):
            sql_query = sql_query.split('```sql')[1]
        if '```' in sql_query:
            sql_query = sql_query.split('```')[0]
        sql_query = sql_query.strip()

        return sql_query
    except Exception as e:
        st.error(f"An error occurred while generating the SQL query: {e}")
        return ""

def handle_follow_up(user_follow_up, table_name, columns, df):
    try:
        # Get the last query from the conversation history
        previous_query = st.session_state.conversation_history[-1]['query'] if st.session_state.conversation_history else None
        
        # Generate synonyms for follow-up question, including the previous question
        synonyms = generate_synonyms(user_follow_up, previous_query)
        sql_query = generate_sql_query(user_follow_up, synonyms, table_name, columns, previous_query)

        # Execute the generated query
        query_result = execute_sql_query(sql_query, df, table_name)

        # Display the follow-up question and result
        st.markdown("### Follow-Up Question:")
        st.markdown(user_follow_up)

        st.markdown("### Generated SQL Query:")
        st.code(sql_query, language='sql')

        st.markdown("### Query Result:")
        st.dataframe(query_result)
        
        return sql_query, query_result  # Return both SQL query and result for further handling
    except Exception as e:
        st.error(f"An error occurred while generating or executing the SQL query for the follow-up: {e}")
        return None, None  # Return None for both if an error occurs

def main():
    st.set_page_config(page_title="SQL Query Generator", page_icon=":robot:", layout="wide")
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #c4aead;
    }
    .stSidebar {
        background-color: #533b4b;
    }
    .stTextInput {
        background-color: #ffe4e1;
    }.stTextArea {
        background-color: #ffe4e1;
    }
    .stButton {
        color: #533b4b;
    }
    </style>
    """, unsafe_allow_html=True
)

    # Initialize session state for memory
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar with sections
    with st.sidebar:
        st.header("Quirkle")
        st.button("AI Chat Helper")
        st.subheader("Templates")
        st.subheader("My Projects")
        st.subheader("Statistics")
        st.subheader("Settings")
        st.subheader("Updates & FAQ")
        st.markdown("---")
        st.markdown("#### Pro Plan")
        st.markdown("Strengthen AI, only $10/month!")

    # Main application interface
    st.markdown("## SQL Query Generator")

    # Display past conversation history first
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        for index, item in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Query {index+1}:** {item['query']}")
            st.markdown("**Generated SQL Query:**")
            st.code(item['response'], language='sql')
            # Display the result from the last query
            if item['result'] is not None:
                st.markdown("**Query Result:**")
                st.dataframe(item['result'])

    # Layout with file uploader always below the conversation history
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        table_name = uploaded_file.name.split('.')[0]  # Using file name as table name
        columns = df.columns.tolist()

        text_input = st.text_area("Enter your Query here in Plain-English:")
        submit = st.button("Generate SQL Query")
        
        st.markdown(f"### Table Name: {table_name}")
        st.markdown(f"### Columns: {', '.join(columns)}")

        if submit:
            with st.spinner("Generating SQL Query..."):
                try:
                    synonyms = generate_synonyms(text_input)
                    sql_query = generate_sql_query(text_input, synonyms, table_name, columns)

                    # Execute the generated query
                    query_result = execute_sql_query(sql_query, df, table_name)

                    # Display the generated SQL query and the result
                    st.markdown("### Generated SQL Query:")
                    st.code(sql_query, language='sql')

                    st.markdown("### Query Result:")
                    st.dataframe(query_result)

                    # Update session state with the new query and response
                    st.session_state.conversation_history.append({
                        'query': text_input,
                        'response': sql_query,
                        'result': query_result
                    })

                except Exception as e:
                    st.error(f"An error occurred while generating or executing the SQL query: {e}")

        # Handle follow-up question input
        follow_up = st.text_area("Enter your Follow-Up Question:")

        follow_up_submit = st.button("Submit Follow-Up Question")

        if follow_up_submit and follow_up:
            with st.spinner("Processing follow-up..."):
                # Get the last query from the conversation history
                previous_query = st.session_state.conversation_history[-1]['query'] if st.session_state.conversation_history else None
                follow_up_sql_query, follow_up_result = handle_follow_up(follow_up, table_name, columns, df)

                # Store follow-up response in session state for conversation history
                if follow_up_result is not None:
                    st.session_state.conversation_history.append({
                        'query': follow_up,
                        'response': follow_up_sql_query,  # Store the actual generated SQL query here
                        'result': follow_up_result
                    })
if __name__ == "__main__":
    main()

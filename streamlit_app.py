# streamlit app creation



# LIBRARIES

import streamlit as st
import plotly.express as px

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import os
import yaml

import pandas as pd
import sqlalchemy as sql
import plotly.io as pio

from MA_agent import lead_scoring_app_agent

# AI SETUP

#os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['OPENAI_API_KEY']

# for streamlit
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Live demo toggle: default first, fallback second.
MODEL_LIST = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-4.1", "gpt-4.1-mini",  "gpt-4o-mini"]

# SQL DATABASE SETUP

PATH_DB = "sqlite:///database/leads_scored.db"

sql_engine = sql.create_engine(PATH_DB)

conn = sql_engine.connect()


# * STREAMLIT APP SETUP ----

st.set_page_config(page_title="Business Intelligence AI Agent ")
st.title("Business Intelligence AI Agent")

st.markdown("""
            I'm a handy business intelligence agent that connects up to the leads_scored.db SQLite database that mimics an ERP System for a company. You can ask me Business Intelligence, Customer Analytics, and Data Visualization Questions. I will report the results. 
            """)

# * model selection

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    MODEL_LIST,
    index=0
)

# * AGENT

app = lead_scoring_app_agent(
    path = PATH_DB,
    model = model_option
)

# * STREAMLIT 

example_questions = st.expander("Try out example questions")

with example_questions:
    """
    Example Questions:
    1. What tables are in the database?
    2. What does the transactions table contain?
    3. What does the products table contain?
    4. What does the leads_scored table contain?
    5. What is the average p1 lead score of leads in the database?
    6. What is the average p1 lead score of leads by member rating in the database?
    7. Calculate the the average p1 lead score of leads by member rating and return a scatter plot with a trendline.
    8. Which 10 customers have the highest p1 probability of purchase who have NOT purchased "Learning Labs Pro - Paid Course"?
    9. What are the top 5 product sales revenue by product name? Make a donut chart. Use suggested price for the sales revenue and a unit quantity of 1 for all transactions.
    10. What are the total sales by month-year? Use suggested price as a proxy for revenue for each transaction and a quantity of 1. Make a chart of sales over time.
    11. What are the total sales by charge_country for the top 10 countries? Make a horizontal bar chart with the charge_country ascending. Use suggested price as a proxy for revenue for each transaction and a quantity of 1.
    """
    
# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Initialize plot storage in session state
if "plots" not in st.session_state:
    st.session_state.plots = []

# Initialize dataframe storage in session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Function to display chat messages including Plotly charts and dataframes
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        msg_text = msg.text
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg_text:
                plot_index = int(msg_text.split("PLOT_INDEX:")[1])
                st.plotly_chart(st.session_state.plots[plot_index], key=f"history_plot_{plot_index}")
            elif "DATAFRAME_INDEX:" in msg_text:
                df_index = int(msg_text.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index], key=f"history_dataframe_{df_index}")
            else:
                st.write(msg_text)

# Render current messages from StreamlitChatMessageHistory
display_chat_history()

if question := st.chat_input("Enter your question here:", key="query_input"):
    with st.spinner("Thinking..."):
        
        st.chat_message("human").write(question)
        msgs.add_user_message(question)
        
        # Run the app
        inputs = {"user_question": question}
        
        error_occured = False
        try: 
            result = app.invoke(inputs)
        except Exception as e:
            error_occured = True
            print(e)
        
        if not error_occured:

            if result['routing_preprocessor_decision'] == 'table':
                # Table was requested
                
                response_text = f"Returning the table...\n\nSQL Query:\n```sql\n{result['sql_query']}\n```"
                
                response_df = pd.DataFrame(result['data'])

                # Store the dataframe and keep its index
                df_index = len(st.session_state.dataframes)
                
                st.session_state.dataframes.append(response_df)

                # Store the response text and dataframe index in the messages
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

                st.chat_message("ai").write(response_text)
                st.dataframe(response_df)
                
            elif result['routing_preprocessor_decision'] == 'chart' and result['chart_plotly_error'] is False:
                # Chart was requested and  produced correctly
                
                response_text = f"Returning the plot...\n\nSQL Query:\n```sql\n{result['sql_query']}\n```"
                
                response_plot = pio.from_json(result["chart_plotly_json"])

                # Store the plot and keep its index
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(response_plot)

                # Store the response text and plot index in the messages
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")

                st.chat_message("ai").write(response_text)
                st.plotly_chart(response_plot)
            else:
                # Chart error occurred, return Table instead
                response_text = f"I apologize. There was an error during the plotting process. Returning the table instead...\n\nSQL Query:\n```sql\n{result['sql_query']}\n```"
                
                df = pd.DataFrame(result['data'])

                # Store the dataframe and keep its index
                df_index = len(st.session_state.dataframes)
                
                st.session_state.dataframes.append(df)

                # Store the response text and dataframe index in the messages
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

                st.chat_message("ai").write(response_text)
                st.dataframe(df)
        else:
            # SQL error occurred
            response_text = f"An error occurred in generating the SQL. I apologize. Please try again or format the question differently and I'll try my best to provide a helpful answer."
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)
            

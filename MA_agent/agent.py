# LIBRARIES
# imported functions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain

from langgraph.graph import END, StateGraph

from pprint import pprint

from typing_extensions import TypedDict

import pandas as pd
import sqlalchemy as sql

import plotly.io as pio

from MA_agent.parsers import SQLOutputParser, PythonOutputParser

# AGENT

def lead_scoring_app_agent(
    path: str = "sqlite:///database/leads_scored.db",
    model: str = "gpt-5.4-mini"
):
    """
    Creates a Business Intelligence Agent that can answer questions, generate SQL, and create charts.
    
    Parameters:
    ----------
    path : str
        The path to the SQLite database.
    model : str
        The OpenAI model to use.
        
    Returns:
    -------
    app : CompiledStateGraph
        The compiled state graph for the Business Intelligence Agent.    
    """    
    
    # Handle case when users want to make a different model than ChatOpenAI
    
    class ChatOpenAINoStop(ChatOpenAI):
        def bind(self, **kwargs):
            kwargs.pop("stop", None)
            return super().bind(**kwargs)

    def get_model_kwargs(model_name: str) -> dict:
        kwargs = {
            "model": model_name,
            "use_responses_api": True,
        }
        if model_name.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": "none"}
        return kwargs

    if isinstance(model, str):
        llm = ChatOpenAINoStop(**get_model_kwargs(model))
        llm_sql_only = ChatOpenAINoStop(**get_model_kwargs(model))
    else:
        llm = model
        llm_sql_only = model

    PATH_DB = path
    
    # * Routing Preprocessor Agent
    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a SQL database agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a SQL translator agent 
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the SQL generator agent. This will be the 'formatted_user_question_sql_only'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the SQL query. If unknown, not specified or 'None' is found, then select 'table'.  
        
        Return JSON with 'formatted_user_question_sql_only' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {initial_question}
        """,
        input_variables=["initial_question"]
    )

    routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

    routing_preprocessor


    # * SQL Agent

    db = SQLDatabase.from_uri(PATH_DB)

    # * New: SQL Output Parser

    prompt_sqlite = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template="""
        You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
        
        Do not use a LIMIT clause with {top_k} unless a user specifies a limit to be returned.
        
        Return SQL in ```sql ``` format.
        
        Only return a single query if possible.
        
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
        
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        
        Pay attention to use date(\'now\') function to get the current date, if the question involves "today".
            
        Only use the following tables:
        {table_info}
        
        Question: {input}'
        """
    )

    sql_generator = (
        create_sql_query_chain(
            llm = llm_sql_only,
            db = db,
            k = int(1e7),
            prompt = prompt_sqlite
        ) 
        | SQLOutputParser() # NEW SQLCodeExtactor
    )


    # * Chart Instructor Agent

    # * NEW: Creates new instructions specifically for the Chart Generator Agent from the User Question
    prompt_chart_instructions = PromptTemplate(
        template="""
        You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting. 
        
        You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart from the data that will be passed to a chart generator agent.
        
        USER QUESTION: {question}
        
        DATA: {data}
        
        Formulate "chart generator instructions" by informing the chart generator of what type of plotly plot to use (e.g. bar, line, scatter, etc) to best represent the data. 
        
        Come up with an informative title from the user's question and data provided. Also provide X and Y axis titles.
        
        Instruct the chart generator to use the following theme colors, sizes, etc:
        
        - Use this color for bars and lines:
            'blue': '#3381ff',
        - Base Font Size: 8.8 (Used for x and y axes tickfont, any annotations, hovertips)
        - Title Font Size: 13.2
        - Line Size: 0.65 (specify these within the xaxis and yaxis dictionaries)
        - Add smoothers or trendlines to scatter plots unless not desired by the user
        - Do not use color_discrete_map (this will result in an error)
        - Hover tip size: 8.8
        
        Return your instructions in the following format:
        CHART GENERATOR INSTRUCTIONS: FILL IN THE INSTRUCTIONS HERE
        
        """,
        input_variables=['question', 'data']
    )

    chart_instructor = prompt_chart_instructions | llm | StrOutputParser()


    # * Chart Generator Agent

    prompt_chart_generator = PromptTemplate(
        template = """
        You are an expert in creating data visualizations and plots using the plotly python library. You must use plotly or plotly.express to produce plots. Your job is to produce python code to generate visualizations.
        
        # IMPORTANT NOTES:
        
        1. Return a single function named, "plot_chart" that ingests a parameter containing "data", and outputs the plotly fig.
        2. Return Python code in ```python ``` format.
        3. Important: Keep the scope of the plot_chart() function local (imports and helper functions inside the main function). This makes it easier to use this function with exec()
        
        CHART INSTRUCTIONS: 
        {chart_instructions}
        
        INPUT DATA: 
        {data}
        
        EXAMPLE FUNCTION CODE TO RETURN (USE THIS FORMAT):
        
        ```python
        def plot_chart(data):
        
            # Import Libraries inside function
            import pandas as pd
            import plotly.express as px
            
            # Create Plot
            fig = px.bar(data, x='Category', y='Value')
            
            return fig
        ```
        
        Important Notes on creating the chart code:
        - Do not use color_discrete_map. This is an invalid property.
        - If bar plot, do not add barnorm='percent' unless user asks for it
        - If bar plot, do not add a trendline. Plotly bar charts do not natively support the trendline.  
        - For line plots, the line width should be updated on traces (example: # Update traces
    fig.update_traces(line=dict(color='#3381ff', width=0.65)))
        - For Bar plots, the default line width is acceptable
        - Super important - Make sure to print(fig_dict)
        - Never return a white background. Use plotly's default theme. A white background can cause issues when rendering in Streamlit.
        - Never a white plot canvas. 
        """,
        input_variables=["chart_instructions", "data"]
    )


    chart_generator = (
        prompt_chart_generator | llm | PythonOutputParser()
    )

    # * LANGGRAPH
    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        user_question: str
        formatted_user_question_sql_only: str
        sql_query : str
        data: dict
        routing_preprocessor_decision: str
        chart_generator_instructions: str
        chart_plotly_code: str
        chart_plotly_json: dict
        chart_plotly_error: bool
        
    def preprocess_routing(state):
        print("---ROUTER---")
        question = state.get("user_question")
        
        # Chart Routing and SQL Prep
        response = routing_preprocessor.invoke({"initial_question": question})
        
        formatted_user_question_sql_only = response['formatted_user_question_sql_only']
        
        routing_preprocessor_decision = response['routing_preprocessor_decision']
        
        return {
            "formatted_user_question_sql_only": formatted_user_question_sql_only,
            "routing_preprocessor_decision": routing_preprocessor_decision,
        }
        


    def generate_sql(state):
        print("---GENERATE SQL---")
        question = state.get("formatted_user_question_sql_only")
        
        # Handle case when formatted_user_question_sql_only is None:
        if question is None:
            question = state.get("user_question")
        
        # Generate SQL
        sql_query = sql_generator.invoke({"question": question})
        
        return {"sql_query": sql_query}


    def convert_dataframe(state):
        print("---CONVERT DATA FRAME---")

        sql_query = state.get("sql_query")
        
        # Generate Data Frame
        sql_engine = sql.create_engine(path)
        conn = sql_engine.connect()
        sql_query_2 = sql_query.rstrip("'")
        df = pd.read_sql(sql_query_2, conn)
        conn.close()
        
        return {"data": df.to_dict(orient="records")}


    def decide_chart_or_table(state):
        print("---DECIDE CHART OR TABLE---")
        return "chart" if state.get('routing_preprocessor_decision') == "chart" else "table"

    def instruct_chart_generator(state):
        print("---INSTRUCT CHART GENERATOR---")
        
        question = state.get("user_question")
        
        data = state.get("data")
        
        # if data is large, sample
        df = pd.DataFrame(data)
        if df.shape[0] > 1000:
            data = df.sample(1000).to_dict(orient="records")
        
        chart_generator_instructions = chart_instructor.invoke({"question": question, "data": data})
        
        return {"chart_generator_instructions": chart_generator_instructions}


    def generate_chart(state):
        print("---GENERATE CHART---")
        
        chart_instructions = state.get("chart_generator_instructions")
        
        data = state.get("data")
        
        # if data is large, sample
        df = pd.DataFrame(data)
        if df.shape[0] > 1000:
            data = df.sample(1000).to_dict(orient="records")
        
        # Generate Chart Python Code
        response = chart_generator.invoke({"chart_instructions": chart_instructions, "data": data})
        
        chart_plotly_error = False
        fig_json = None
        if "error" in response[:40].lower():
            chart_plotly_error = True
        else:
            try:
                # Create dictionaries to hold the local and global variables
                local_vars = {}
                global_vars = {}
                
                exec(response, global_vars, local_vars)
                
                plot_chart = local_vars.get("plot_chart")

                fig = plot_chart(df)
                
                fig_json = pio.to_json(fig)
            except:
                chart_plotly_error = True
            
        return {
            "chart_plotly_code": response, 
            "chart_plotly_json": fig_json, 
            "chart_plotly_error": chart_plotly_error,
        }
        
        
    def state_printer(state):
        """print the state"""
        print("---STATE PRINTER---")
        print(f"User Question: {state['user_question']}")
        print(f"Formatted Question (SQL): {state['formatted_user_question_sql_only']}")
        print(f"SQL Query: \n{state['sql_query']}\n")
        print(f"Data: \n{pd.DataFrame(state['data'])}\n")
        print(f"Chart or Table: {state['routing_preprocessor_decision']}")
        
        if state['routing_preprocessor_decision'] == "chart":
            print(f"Chart Code: \n{pprint(state['chart_plotly_code'])}")
            print(f"Chart Error: {state['chart_plotly_error']}")
        

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_routing", preprocess_routing)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("convert_dataframe", convert_dataframe)
    workflow.add_node("instruct_chart_generator", instruct_chart_generator)
    workflow.add_node("generate_chart", generate_chart)
    workflow.add_node("state_printer", state_printer)

    workflow.set_entry_point("preprocess_routing")
    workflow.add_edge("preprocess_routing", "generate_sql")
    workflow.add_edge("generate_sql", "convert_dataframe")

    workflow.add_conditional_edges(
        "convert_dataframe", 
        decide_chart_or_table,
        {
            # Result : Step Name To Go To
            "chart":"instruct_chart_generator", # Path Chart
            "table":"state_printer", # Path State Printer
            # "error":"generate_sql", # Path State Printer
        }
    )

    workflow.add_edge("instruct_chart_generator", "generate_chart")
    workflow.add_edge("generate_chart", "state_printer")
    workflow.add_edge("state_printer", END)

    app = workflow.compile()

    return app

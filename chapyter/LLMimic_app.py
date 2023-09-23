import streamlit as st
from streamlit_chat import message

import pandas as pd
import openai
import matplotlib
import scipy
import boto3
import os
import time

from tabulate import tabulate

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessage,
    AIMessage,
    HumanMessage,
)

DATABASE = 'mimiciii'
S3_BUCKET = 's3://emmett-athena-bucket/'
AWS_REGION = 'us-east-1'

LLM = ChatOpenAI(model_name="gpt-4", max_tokens=2000)
# LLM = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=2000)

client = boto3.client('athena', region_name=AWS_REGION)

if 'df' not in st.session_state:
    st.session_state.df = False

def dataframe_to_markdown(df):
    return tabulate(df, headers='keys', tablefmt='pipe')


def extract_value(cell):
    """Extracts value from a cell returned by Athena. Supports multiple types."""
    if 'VarCharValue' in cell:
        return cell['VarCharValue']
    elif 'LongValue' in cell:
        return cell['LongValue']
    elif 'DoubleValue' in cell:
        return cell['DoubleValue']
    else:
        return None  # or you can return a default value or throw an exception

def sql_query_to_athena_df(query):
    """
    Takes in a string query, and returns df from Athena
    """
    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': DATABASE
        },
        ResultConfiguration={
            'OutputLocation': S3_BUCKET
        }
    )

    query_execution_id = response['QueryExecutionId']

    while True:
        response = client.get_query_execution(QueryExecutionId=query_execution_id)
        if response['QueryExecution']['Status']['State'] in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(5)

    df = pd.DataFrame()

    if response['QueryExecution']['Status']['State'] == 'SUCCEEDED':
        # Initialize pagination token
        next_token = None
        
    while True:
        # print("PAGINATING")
        if next_token:
            result_data = client.get_query_results(QueryExecutionId=query_execution_id, NextToken=next_token)
        else:
            result_data = client.get_query_results(QueryExecutionId=query_execution_id)
        
        headers = [i['Name'] for i in result_data['ResultSet']['ResultSetMetadata']['ColumnInfo']]
        rows = [[extract_value(cell) for cell in row['Data']] for row in result_data['ResultSet']['Rows'][1:]]
        
        # Append the rows to the DataFrame using pd.concat
        new_df = pd.DataFrame(rows, columns=headers)
        df = pd.concat([df, new_df], ignore_index=True)
        
        next_token = result_data.get('NextToken')
        if not next_token:
            break
    else:
        print("Query failed!")

    return df, query



def chat_conversation(user_input):
    if 'conversation' not in st.session_state:
        st.session_state.conversation = [SystemMessage(content="I'm a helpful AI assistant. Whenever the user asks for python code, and its for tabular data analysis, I will assume they have a dataframe 'df'. ")]

    human_message_prompt = HumanMessage(content=user_input, avatar_stype='adventurer-neutral')
    st.session_state.conversation.append(human_message_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(st.session_state.conversation)
    chain = LLMChain(llm=LLM, prompt=chat_prompt)
    llm_output = chain.run(input_language="English")

    ai_message_prompt = AIMessage(content=llm_output)
    st.session_state.conversation.append(ai_message_prompt)

    return llm_output

def query_llm(llm_prompt, sys_prompt):
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": llm_prompt}
        ],
        max_tokens=1000,
        temperature=0.1,
    )
    response = response["choices"][0]["message"]["content"]
    return response

def main():
    st.title("LLMimic-III")

    if 'generated' not in st.session_state:
        st.session_state.generated = []

    user_input = st.text_input("Enter something:")

    submit_button = st.button("Send Message")
    if submit_button:
        st.session_state.generated.append(["Human", user_input])
        AI_language_response = chat_conversation(user_input)
        st.session_state.generated.append(["AI", AI_language_response])

        sys_prompt = "You are an expert coder. If the text/code sent to you contains a SQL query, respond with only the SQL query found, which is directly executable in Athena. Don't return the query in the form of a triple string. Change things that look like 'mimic.mimiciii.patients' to simply 'patients'. Otherwise respond only 'NO SQL FOUND'."
        contains_SQL = query_llm(AI_language_response, sys_prompt) 
        if contains_SQL != "NO SQL FOUND": #If the response has SQL in it, execute it, get df and store it
            df = sql_query_to_athena_df(contains_SQL)

            st.session_state.df = df  # Save the DataFrame to session state

            content_to_add_to_conversation = "Here is your retrieved table:"
            st.session_state.conversation.append(AIMessage(content=content_to_add_to_conversation))
            st.session_state.generated.append(["AI", content_to_add_to_conversation])

            content_to_add_to_conversation = dataframe_to_markdown(df.head(3))        
            st.session_state.conversation.append(AIMessage(content=content_to_add_to_conversation))
            st.session_state.generated.append(["AI", content_to_add_to_conversation])

        sys_prompt = """You are an expert coder. 
        If the text/code sent to you contains python code, respond with only the python code found. 
        Don't assume the datatypes in the table are anything, always check them and change if needed. 
        No comments. Should be directly executable. Assume we already have the dataframe called 'df', if one is needed. 
        MUST return a variable called 'answer' Otherwise respond only 'NO PYTHON CODE FOUND'.
        """

        contains_python = query_llm(AI_language_response, sys_prompt) 
        if contains_python != "NO PYTHON CODE FOUND" and isinstance(st.session_state.df, pd.DataFrame):
            print("GOT PYTHON CODE", contains_python)

            #Now that we got the code, lets analyze our df, need to retrieve from session state dict            
            #And then we will execute the code on the fly
            try:
                df = st.session_state.df
                context = {'df': df}
                exec(contains_python, globals(), context)
                answer = context.get('answer', None)

            except:
                dont_have_df = True
                # st.session_state.df = None
                exec(contains_python, globals())
                answer = context.get('answer')


            print("From python analysis, got answer:", answer, type(answer))

            code_exec_answer_human_prompt = f"After running the code, I got the answer {answer}. As tersely as possible, can summarize the numbers?"
            AI_language_response = chat_conversation(code_exec_answer_human_prompt)

            
            ai_message_prompt = AIMessage(content=code_exec_answer_human_prompt)
            st.session_state.conversation.append(ai_message_prompt)
            st.session_state.generated.append(["AI", AI_language_response])



    if len(st.session_state.generated) > 0:
        print("GOING TO UNPACK", st.session_state.generated)
        for i, persona_and_message in enumerate(st.session_state.generated[::-1]):
            
            persona, history_message = persona_and_message

            unique_key = f"message_{i}_{persona}"

            is_table = False
            if isinstance(history_message, pd.DataFrame):
                is_table = True

            if persona == "AI":
                message(history_message, key=unique_key, is_table = is_table, allow_html=True, logo="https://i.imgur.com/zurTjqc.png")  # shows the AI's response on the left
            else:
                message(history_message, is_user=True, key=unique_key)  # Shows user's message on the right.

if __name__ == "__main__":
    main()

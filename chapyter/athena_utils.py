import pandas as pd
import openai
import boto3
import time
import matplotlib.pyplot as plt

from tabulate import tabulate

from langchain.chat_models import ChatOpenAI


DATABASE = 'mimiciii'
S3_BUCKET = 's3://emmett-athena-bucket/'
AWS_REGION = 'us-east-1'


LLM = ChatOpenAI(model_name="gpt-4", max_tokens=2000)
# LLM = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=2000)


client = boto3.client('athena', region_name=AWS_REGION)


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

    #Polls Athena until we finally get a response!
    query_execution_id = response['QueryExecutionId']
    while True:
        response = client.get_query_execution(QueryExecutionId=query_execution_id)
        if response['QueryExecution']['Status']['State'] in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(5)

    df = pd.DataFrame()

    if response['QueryExecution']['Status']['State'] == 'SUCCEEDED':
        next_token = None
    else:
        state = response['QueryExecution']['Status']['State']
        print(f"Query failed! State: {state}")
        
        # Log the reason for query failure if available
        if 'StateChangeReason' in response['QueryExecution']['Status']:
            reason = response['QueryExecution']['Status']['StateChangeReason']
            print(f"Reason: {reason}")
        
        return False, reason

        
    #Paginates to get all results from Athena query, and collect them
    while True:
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

    return df, query


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
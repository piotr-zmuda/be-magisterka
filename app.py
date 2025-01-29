from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from transformers import pipeline
import streamlit as st
import pandas as pd
import mysql.connector
import io
import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe
from pandasai.connectors import MySQLConnector

app = Flask(__name__)

model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3.1:latest"
)
def get_normal_response_from_Llama(template):
    messages = [{
    "role": "user",
    "content": (
        template
    )
    }]
    outputs = pipe( messages, max_new_tokens=256, do_sample=False)
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(assistant_response)
    return assistant_response
def get_db_response_from_Llama(db_data, user_query):
    messages = [{
    "role": "user",
    "content": (
        "create nice answer on this question: "+user_query +" , based on this data: "+db_data 
    )
    }]
    outputs = pipe( messages, max_new_tokens=256, do_sample=False)
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(assistant_response)
    return assistant_response

def init_database(user:str, password: str, host:str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


# Example: Restricting CORS to specific origins and headers
CORS(app, resources={r"/api/*": {"origins": "http://localhost:4200"}}, headers=['Content-Type'])

def is_database_query(user_input):
    # Use the classifier to predict the intent of the user input
    result = classifier(user_input)
    # Check if the predicted label indicates a database query
    # Adjust based on the labels your model uses
    return result[0]['label'] == 'LABEL_FOR_DATABASE_QUERY'

def get_sql_chain(db, prompt, llm):

    def get_rows_2(_):
        query = "SELECT * FROM  PRODUCT;"  # Adjust query based on your database type (MySQL, PostgreSQL, etc.)
        cursor.execute(query)
        rows = cursor.fetchall()
        return jsonify(rows)
    def get_columns(_):
        query = "DESCRIBE PRODUCT;"  # Adjust query based on your database type (MySQL, PostgreSQL, etc.)
        cursor.execute(query)
        rows = cursor.fetchall()

        return jsonify(rows)
    def get_schema(_):
        print(db.get_table_info())
        return db.get_table_info()
    return(
        RunnablePassthrough.assign(columns=get_columns)
        | prompt 
        | llm
        | StrOutputParser()
    )


@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        "message": "Hello from Flask!",
        "items": [1, 2, 3, 4, 5]
    }
    return jsonify(data)

@app.route('/api/products', methods=['GET'])
def get_products():
    data = cursor.execute("SELECT * FROM PRODUCT;")
    rows = cursor.fetchall()
    thing_to_add = ""
    # Print the results
    for row in rows:
        thing_to_add+=str(row)+"\n"
    return jsonify(thing_to_add);  

@app.route('/api/generateAI', methods=['POST'])
def get_response_from_ai():
    user_query = request.form.get('user_query')
    chat_history = request.form.get('chat_history')
    template = request.form.get('template')
    llama_query = request.form.get('llama_query')
    file = request.files.get('file')

    if chat_history:
            template = template.replace("{chat_history}", chat_history)
    if user_query:
                template = template.replace("{user_query}", user_query)
    words_to_find = ["shop", "store"]
    data = pd.read_csv(file)
    print(model)
    try:
       # Process the DataFrame with SmartDataframe
       print("GET DF")
       df = SmartDataframe(data, config={"llm": model})
       print("GET RESULT")
       result = df.chat(template)
       print("RETURN RESULT")
       print(result)
       return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#sql generator methods

def read_csv(file):
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    csvreader = csv.reader(stream)
    
    # Assuming the first row contains column names
    columns = next(csvreader)
    
    data = []
    for row in csvreader:
        data.append(row)
    return columns, data

def generate_sql_insert(table_name, columns, data):
    sql_commands = []
    columns = [column.replace(" ", "_") for column in columns]
    print(columns)
    for row in data:
        # Generate SQL INSERT command for each row
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s']*len(columns))})"
        sql_commands.append((sql, row))
    return sql_commands


def add_columns_to_table(cursor, table_name, new_columns):
    new_columns = [column.replace(" ", "_") for column in new_columns]
    for column in new_columns:
        try:
            cursor.execute(f"ALTER TABLE {table_name} MODIFY COLUMN {column} VARCHAR(1500)")  # Assuming VARCHAR(255) data type, adjust as needed
            print(f"Added column {column} to table {table_name}")
        except mysql.connector.Error as err:
            print("Error:", err)

@app.route('/api/generateCsvAnswer', methods=['POST'])
def generateCsvAnswer():
    user_query = request.form.get('user_query')
    try:

        # Process the DataFrame with SmartDataframe
        df = SmartDataframe(mysql_connector, config={"llm": model})
        result = df.chat(user_query)
        # Check if the result is a DataFrame
        if isinstance(result, pd.DataFrame):
            # Convert DataFrame to JSON format
            result_json = result.to_json(orient='records')
            return jsonify(result_json)
        else:
            return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500    


@app.route('/api/generateSqlScript', methods=['POST'])
def generateTable():
    file = request.files['file']
    print(file)
    print("asdasdasd")
    table_name = 'PRODUCT'
    new_columns, data = read_csv(file=file)

    cursor = mydb.cursor()




    # Add new columns to the table

    # Add new columns to the table
    add_columns_to_table(cursor, table_name, new_columns)

    # Generate SQL INSERT commands
    sql_commands = generate_sql_insert(table_name, new_columns, data)

    # Execute SQL INSERT commands
    for sql_command in sql_commands:
        print(sql_command)
        cursor.execute(sql_command[0], sql_command[1])

    mydb.commit()
    mydb.close()  

    return "Succesfully added csv to docker"

if __name__ == '__main__':
    app.run(debug=True)



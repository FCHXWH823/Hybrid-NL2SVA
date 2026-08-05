import os
import json
import sys
import yaml
from openai import OpenAI

with open("Src/Config.yml") as file:
    config = yaml.safe_load(file)
# Load your PDFs
OpenAI_API_Key = config["Openai_API_Key"]
DeepSeek_API_Key = config["DeepSeek_API_Key"]
Model_Name = config["Model_Name"]

client = OpenAI(
        api_key=OpenAI_API_Key,
)


import re

def parse_numbered_output(text):
    """
    Parses a numbered list in the format [n]. text and extracts each item.

    Args:
        text (str): The multiline string output.

    Returns:
        List[str]: A list of extracted parts in order.
    """
    # Pattern to match [1]. ..., [2]. ..., etc.
    lines = text.splitlines()
    outputs = []
    for line in lines:
        # get the content 1. ...
        pattern = r"^\d+\.\s+(.*)"
        match = re.match(pattern, line)
        if match:
            outputs.append(match.group(1).strip())
    return outputs


# use embedding=OpenAIEmbeddings(openai_api_key=OpenAI_API_Key) to get the embedding of the expalanations
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings


def extract_keywords(nl_sva):
    prompt_split = f"Split the following sentence\n{nl_sva}\n into multiple parts, each representing an operation over a single signal or group of signals. Present the output as a numbered list in the following format:\n1. <First operation>\n2. <Second operation>\n3. <Third operation>\n...\n"
    completion = client.chat.completions.create(
                model= Model_Name,
                messages=[
                    {"role": "system", "content": "You are a helpful bot to split a given sentence into multiple parts."},
                    {"role": "user", "content": prompt_split}
                ]
    )
    
    # print("Split Results: ", completion.choices[0].message.content)

    parsed = parse_numbered_output(completion.choices[0].message.content)
    return parsed

def extract_keywords(nl_sva):
    prompt_split = f"Split the following sentence\n{nl_sva}\n into multiple parts. Each part should represent either: \n1. an operation on a single signal or a group of related signals;\n 2. or a temporal keyword or phrase commonly used in formal specification languages.\n Present the output as a numbered list in the following format:\n1. <First operation>\n2. <Second operation>\n3. <Third operation>\n...\n"
    completion = client.chat.completions.create(
                model= Model_Name,
                messages=[
                    {"role": "system", "content": "You are a helpful bot to split a given sentence into multiple parts."},
                    {"role": "user", "content": prompt_split}
                ]
    )
    
    # print("Split Results: ", completion.choices[0].message.content)

    parsed = parse_numbered_output(completion.choices[0].message.content)
    return parsed


def extract_related_operators_of_keyword(keywords):
    operators = set()
    # read the json file and print the operators and explanations
    with open("operators.json","r") as file:
        data = json.load(file)
        ops = []
        ops_explanation = ""
        for key, value in data.items():
            ops.append(key)
            ops_explanation += f"{key}: {value}\n"
            
        
        # nl_sva = "when the third bit of grant output is asserted and in the previous clock cycle the arbitration type selector signal was equal to 1, then in the previous clock cycle the third bit of request input signal must be asserted and the first and second bits of request input signal must not be asserted from the current clock cycle"
        
    for i, keyword in enumerate(keywords, 1):
        prompt = f"Given a set of systemverilog assertion operatosrs and their explanations as follows:\n {ops_explanation}\n Please extract the most relevant operator from the natural language input \n`{keyword}`\n, but do not return anything if no relevant operator exists.\n"
        completion = client.chat.completions.create(
                    model= Model_Name,
                    messages=[
                        {"role": "system", "content": "You are a helpful bot to extract the relevant systemverilog assertion operator from a given list."},
                        {"role": "user", "content": prompt}
                    ]
        )

        for op in ops:
            if op in completion.choices[0].message.content:
                operators.add(op)
        
        ops_explanations = []
        for op in operators:
            ops_explanations.append(f"`{op}`: {data[op]}")
    return ops_explanations

    

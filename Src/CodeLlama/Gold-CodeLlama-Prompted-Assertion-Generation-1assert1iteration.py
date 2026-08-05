from openai import OpenAI
import pandas as pd
import csv
import random
import yaml
import os
import json
import re
import yaml
from langchain_ollama import ChatOllama
#client = OpenAI()
from langchain_openai import ChatOpenAI
from json import JSONDecodeError
from docker_vcs import *

def remove_last_endmodule(verilog_code):
    lines = verilog_code.strip().split("\n")
    
    # Find the last occurrence of "endmodule"
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "endmodule":
            del lines[i]
            break  # Remove only the last occurrence

    return "\n".join(lines)

def init_prompt_template_completion(model_id,code):
    return client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful bot that provides a brief explanation of the input code."},
            {"role": "user", "content": "Given input code snippet as below: \n" + code + "\n Please give a brief explanation of the code. \n"}
        ]
    )

with open("Src/Config.yml") as file:
    config = yaml.safe_load(file)

OpenAI_API_Key = config["Openai_API_Key"]
Model_Name = config["Model_Name"]
Excute_Folder = config["Excute_Folder"]
CodeLlama_Model_Name = config["CodeLlama_Model_Name"]
PostProcess_Model_Name = config["PostProcess_Model_Name"]

OpenAI_client = ChatOpenAI(
    model=PostProcess_Model_Name,
    api_key=OpenAI_API_Key
)

client = ChatOllama(
    model = CodeLlama_Model_Name,
    temperature = 0.6,
    num_predict = 256,
    top_p=1
)


def assertion_checker_prompt(llm_response, assertion_format):
    return f'''
    Please correct the following systemverilog assertion if there exist some syntax errors in it, such as unmatched parentheses:
    {llm_response}
    Please output the corrected assertion STRICTLY in the following format:
    {assertion_format}
    '''

def assertion_format_prompt(llm_response, assertion_format):
    return f'''
    Please rewrite the following systemverilog assertion without changing its functionality:
    {llm_response}
    to STRICTLY follow the format:
    {assertion_format}
    Please ONLY output the rewritten one.  
    '''

def extract_last_module(verilog_code: str) -> str:
    """
    Extract the last Verilog module from the given Verilog code string.
    
    Parameters:
        verilog_code (str): A string containing the Verilog code.
    
    Returns:
        str: The last module found in the code, or an empty string if no module is found.
    """
    # Use a regex pattern with non-greedy matching to capture each module block.
    # The pattern looks for a word boundary followed by 'module', then matches until
    # the first occurrence of 'endmodule' (also at a word boundary).
    pattern = r'\b(module\b.*?\bendmodule\b)'
    
    # Use DOTALL so that the dot (.) matches newline characters.
    modules = re.findall(pattern, verilog_code, flags=re.DOTALL)
    
    if modules:
        return modules[-1].strip()
    else:
        return ""

def extract_prerequist_of_assertions(verilog_code_w_assertions:str, verilog_code_wo_assertions:str, num_assertions: int):
    lines_w_assertions = extract_last_module(verilog_code_w_assertions).strip().split("\n")

    lines_wo_assertions = extract_last_module(verilog_code_wo_assertions).strip().split("\n")

    lines_assertions = lines_w_assertions[len(lines_wo_assertions):len(lines_w_assertions)-1]

    # lines_prerequist = lines_assertions[:-num_assertions]
    lines_assertions.append("// above are golden assertions\n\n")
    return lines_assertions

def golden_check(explanation, logic_expression):

    checking_str = ""
    ops = []
    with open("sva_property_sequence_operators.jsonl","r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            ops.append(json.loads(line)) 
    
    if "|=>" in logic_expression or "|->" in logic_expression:
        checking_str += '''"Op": "|->", "Usage": "sequence_expr |-> property_expr",  "Explanation": "for every match of the sequence_expr beginning at the start point, the evaluation of the property_expr beginning in the current clock cycle at the end point of the match succeeds and returns true."\n'''
        checking_str += '''"Op": "|=>", "Usage": "sequence_expr |=> property_expr",  "Explanation": "for every match of the sequence_expr beginning at the start point, the evaluation of the property_expr beginning at the next clock cyle after the end point of the match succeeds and returns true."\n'''

    for op in ops:
        if re.search(op["Op_Regex"], logic_expression):
            checking_str += f'''"Op": "{op['Op']}", "Usage": "{op['Usage']}",  "Explanation": "{op['Explanation']}"\n'''
            # retrieved_doc = code_retriever.invoke(f"`{op}`: {operators[op]}")                    
            # for doc in retrieved_doc:
            #     checking_str += doc.page_content + "\n\n"
            # checking_str += "\n"    


    # checking_str += llm_response_explain
    # print(f"checking_str: {checking_str}")
    completion = client.invoke(input=[
        {"role": "system", "content": "You are a helpful bot that answer some questions about SystemVerilog."},
        {"role": "user", "content": f"Given the desired explanation\n{explanation},\n please check whether the systemverilog assertion {logic_expression} operates with the correct logic and the same timing (i.e., clock cycle).\n The relevant context of the used operators are given:\n {checking_str}.\n If there exists, please modify it into a new systemverilog assertion and output the new assertion.\n"}
    ])
    
    # print(completion.choices[0].message.content)
    return completion.content


def assertion_generation(code, assertion, details):
    explanation = details.get("Assertion Explaination", "No explanation provided")
    signals = details.get("Signals", "No signals provided")
    # clk_condition = "" if details.get("clock signal condition") is "none" else details.get("clock signal condition")
    # reset_condition = "" if details.get("disable condition") is "none" else details.get("disable condition")
    
    assertion_format = f"assert property (ONLY logical expression WITHOUT clock signal condition @(posedge clock) and WITHOUT disable condition disable iff(...));\n Note that the output assrtion must be end with semicolon."
    

    prompt = f"Given the used signals \n{signals}\n Please generate the systemverilog assertion for the following natural language property explanation:\n{explanation}.\nThe output format should STRICTLY follow :\n{assertion_format}\nWITHOUT other things."

    completion = client.invoke(input=[
        {"role": "system", "content": "You are a helpful bot that generate the assertion satisfying some requirements for a given verilog code."},
        {"role": "user", "content": prompt}
    ])

    llm_response = completion.content

    # completion = client.chat.completions.create(
    # model=model_name,
    # messages=[
    #     {"role": "system", "content": "You are a helpful bot that generate the assertion satisfying some requirements for a given verilog code."},
    #     {"role": "user", "content": prompt}
    # ]
    # )

    # assertion checker
    # nItChecker = 3
    # for it in range(nItChecker):
    #     checker_prompt = assertion_checker_prompt(completion.content,assertion_format)
    #     completion = client.invoke(input=[
    #         {"role": "system", "content": "You are a helpful bot that correct the syntax errors of systemverilog assertion if there exists."},
    #         {"role": "user", "content": checker_prompt}
    #     ])

    completion = OpenAI_client.invoke(
    input=[
        {"role": "user", "content": f"Please output the final generated SystemVerilog assertion following the format:\n{assertion_format}\nfrom the following text:\n" + llm_response+ "\nPlease double-check the signals used in the assertion are from the given Verilog code:\n" + code}
    ]
    )
    sva = completion.content.strip()

    checked_output = golden_check(explanation, sva)
    completion = OpenAI_client.invoke(
    input=[
        {"role": "user", "content": f"Please output the final generated SystemVerilog assertion following the format:\n{assertion_format}\nfrom the following text:\n" + checked_output+ "\nPlease double-check the signals used in the assertion are from the given Verilog code:\n" + code}
    ]
    )
    modified_sva = completion
    return modified_sva

with open("Evaluation/Dataset/Ripple_Carry_Adder/explanation.json") as jsonfile:
    data = json.load(jsonfile)

# NYU CCS's key
model_name = Model_Name
checker_model_name = "o3-mini"

eval_num = 100
with open('Results/Openai-4o-mini-Prompted-Assertion-Generation-Results-for-New-Dataset.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Master Module','Code','golden_assertions','llm_assertions'])
    for folder in os.listdir("Evaluation/Dataset/"):
        if Excute_Folder != 'ALL_DESIGNS' and Excute_Folder not in folder:
            continue
        # folder = "BusArbiter"
        leaf_sv_files = []
        folder_path = os.path.join("Evaluation/Dataset/",folder)
        if os.path.isdir(folder_path):
            with open(folder_path+"/"+folder+".sv","r") as file:
                code = file.read()
            with open(folder_path+"/explanation.json") as file:
                # explanation_origin = file.read()
                explanation_json = json.load(file)

            with open(folder_path+"/explanation.json") as file:
                explanation_origin = file.read()
            
            i = 0

            llm_responses = []
            for assertion, details in explanation_json.items():
                if "Assertion" not in assertion:
                    leaf_sv_files = details
                    continue
                i += 1
                while True:
                    completion = assertion_generation(code, assertion, details)
                    match = re.search(r'assert property\s*\(\s*(.*?)\s*\)\s*;', completion.content, re.DOTALL)
                    if match is None:
                        continue
                    matched_str = str(match.group(0))
                    matched_str = matched_str.replace("\n"," ")
                    test_str = "{\n"
                    test_str += f"\"Assertion {i}\": \"{matched_str}\""
                    test_str += "\n}"
                    try:
                        json_test_str = json.loads(test_str)
                    except JSONDecodeError:
                        continue
                    else:
                        llm_responses.append(f"\"Assertion {i}\": \"{matched_str}\"") 
                        break

            llm_response = "{\n"
            for i in range(len(llm_responses)-1):
                llm_response += llm_responses[i]+",\n"
            llm_response +=llm_responses[-1]+"\n}"
            csv_writer.writerow([folder,code,explanation_origin,llm_response])


            if config["JasperGold_VERIFY"] == 1:
                llm_assertions = json.loads(llm_response)
                clk_conditions = []
                disable_conditions = []
                golden_logic_expressions = []
                llm_logic_expressions = []

                for assertion, details in explanation_json.items():
                    if "Assertion" not in assertion:
                        continue
                    clk_condition = "" if details.get("clock signal condition") == "none" else details.get("clock signal condition")
                    disable_condition = "" if details.get("disable condition") == "none" else details.get("disable condition")
                    logic_expression = details.get("logical expression")
                    clk_conditions.append(clk_condition)
                    disable_conditions.append(disable_condition)
                    golden_logic_expressions.append(logic_expression)
                    
                for assertion, details in llm_assertions.items():
                    match = re.search(r'assert property\s*\(\s*(.*?)\s*\)\s*;', details)
                    llm_logic_expressions.append(match.group(1) if match else "")
                
                combine_assertions = []
                with open(folder_path+"/"+folder+".sv","r") as file:
                    verilog_code_wo_assertions = file.read()
                with open(folder_path+"/"+folder+"_assertion.sv","r") as file:
                    verilog_code_w_assertions = file.read()
                # combine_assertions = extract_prerequist_of_assertions(verilog_code_w_assertions,verilog_code_wo_assertions,len(clk_conditions))

                if len(leaf_sv_files):
                    leaf_file_name = leaf_sv_files[0]
                    with open(folder_path+"/"+leaf_sv_files[0]+".v","r") as file:
                        verilog_leaf_sv = file.read()
                else:
                    verilog_leaf_sv = ""
                    leaf_file_name = ""

                for i in range(len(clk_conditions)):
                    combine_assertion = f"assert property ({clk_conditions[i]} {disable_conditions[i]} ({llm_logic_expressions[i]}));" 
                    sc_detect = vcs_process(verilog_leaf_sv, verilog_code_wo_assertions, combine_assertion, leaf_file_name, f"{folder}_{i+1}_llm")
                    if sc_detect:
                        combine_assertion = f"// assert property ({clk_conditions[i]} {disable_conditions[i]} ({llm_logic_expressions[i]}));"     
                        combine_assertions.append(combine_assertion)
                        combine_assertion = f"// assert property ({clk_conditions[i]} {disable_conditions[i]} ({golden_logic_expressions[i]}) iff ({llm_logic_expressions[i]}));" 
                        combine_assertions.append(combine_assertion)
                    else:
                        combine_assertion = f"assert property ({clk_conditions[i]} {disable_conditions[i]} ({llm_logic_expressions[i]}));"     
                        combine_assertions.append(combine_assertion)
                        combine_assertion = f"assert property ({clk_conditions[i]} {disable_conditions[i]} ({golden_logic_expressions[i]}) iff ({llm_logic_expressions[i]}));" 
                        combine_assertions.append(combine_assertion)
                    

                processed_code = remove_last_endmodule(verilog_code_w_assertions)
                processed_code += "\n\n"
                for assertion in combine_assertions:
                    processed_code += assertion+"\n"
                processed_code += "\nendmodule\n"

                with open(folder_path+"/"+folder+f"_Gold-{CodeLlama_Model_Name}.sv","w") as file:
                    file.write(processed_code)
                
                print(f"====================={folder} finished=====================")





                



            
            

# with open('Results/Openai-4o-mini-Prompted-Assertion-Generation-Results.csv', 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(['code','HumanExplanation','pure code','prompt','llm_response'])
#     for id in range(len(df)):
#         code = df.iloc[id]['code']
#         humanexplanation = df.iloc[id]['HumanExplanation']
#         purecode = df.iloc[id]['pure code']
#         prompt = f"Given Verilog code snippet as below: \n{purecode}\n Please generate a rewritten version of it, which contains several useful assertions and each of them follows a description as follows:{humanexplanation}\n"


#         #print("1---------------------------------------------------------------")
#         completion = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": "You are a helpful bot that generate some useful assertions for a given verilog code."},
#             {"role": "user", "content": prompt}
#         ]
#         )

#         llm_response = completion.choices[0].message.content

#         csv_writer.writerow([code,humanexplanation,purecode,prompt,llm_response])
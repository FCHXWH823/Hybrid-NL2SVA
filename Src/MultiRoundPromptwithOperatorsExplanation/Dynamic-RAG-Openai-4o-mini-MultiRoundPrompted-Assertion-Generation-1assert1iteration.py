import os
import sys

# This system's built-in sqlite3 (3.34.1) is below Chroma's minimum (3.35.0);
# swap in pysqlite3-binary's bundled modern build before chromadb (imported
# transitively by extract_keywords.py's own Chroma import just below, and
# again by rag_database.py) checks the version. Idempotent, so having this
# patch in both this file and rag_database.py is safe -- it just needs to
# run before the *first* chromadb import in either possible order.
# See https://docs.trychroma.com/troubleshooting#sqlite
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

import yaml
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import csv
import re
from openai import OpenAI
from extract_keywords import *

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sva_tree"))
from explanation_merge_tree import build_and_render_explanation_merge_tree

with open("Src/Config.yml") as file:
    config = yaml.safe_load(file)
# Load your PDFs
# PDF_Name = "CernyDudani-SVA- The Power of Assertions in SystemVerilog"
PDF_Name = config["PDF_Name"]
PDF_Txt = config["PDF_Txt"]
OpenAI_API_Key = config["Openai_API_Key"]
Folder_Name = f"Book1-{PDF_Name}"
Model_Name = config["Model_Name"]
Excute_Folder = config["Excute_Folder"]

client = OpenAI(
        api_key=OpenAI_API_Key
)

# sva_temporal_operators.json (38 entries, includes strong/weak) replaces the
# older operators.json (11 entries) as the sole operator table -- used for
# HybridRetrieval's keyword-guided path, SOR rechecking (merge-tree + the
# final recheck completion below), and this generation pipeline's own
# fallback glossary.
with open("sva_temporal_operators.json", "r") as file:
    _operators = json.load(file)
operator_context = "\n".join(
    f"{op} ({entry['type']}): {entry['natural_langage_explanation']} Example: {entry['example_usgae']}"
    for op, entry in _operators.items()
)

from rag_database import build_rag_system

code_store = build_rag_system(PDF_Txt, OpenAI_API_Key)

code_retriever = code_store.as_retriever()

# prompt
system_prompt = (
    "You are a helpful bot that generate the assertion satisfying some requirements for a given verilog code."
    "Use the following pieces of retrieved context to help answer the question. "
    "\n\n"
    "{keywords_explaination}"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","Given Verilog code snippet as below: \n{code}\n Please generate such a systemverilog assertion for it following the description:{input}. Ensure the syntax correctness and the used signals should be from the verilog code.\nThe output format should STRICTLY follow :\n{assertion_format}\nWITHOUT other things."),
    ]
)

system_prompt_checker = (
    "You are a helpful bot that check the syntax correctness of the given assertion and corret it if there exist syntaxs error."
    "Use the following pieces of retrieved context to help answer the question. "
    "\n\n"
    "{context}"
)
prompt_checker = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_checker),
        ("human","{input}"),
    ]
)

# retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA

llm = ChatOpenAI(
    model=Model_Name,
    # model="o3-mini",
    api_key=OpenAI_API_Key
    )

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(code_retriever,question_answer_chain)

question_answer_chain_checker = create_stuff_documents_chain(llm,prompt_checker)
rag_chain_checker = create_retrieval_chain(code_retriever,question_answer_chain_checker)


# llm_response = rag_chain.invoke({"input":question})

# llm_response

def assertion_checker_prompt(llm_response, assertion_format):
    return f'''
    Please correct the following systemverilog assertion if there exist some syntax errors in it, such as unmatched parentheses:
    {llm_response}
    Please output the corrected assertion STRICTLY in the following format:
    {assertion_format}
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
    lines_assertions.append("// above are golden assertions")
    return lines_assertions

def remove_last_endmodule(verilog_code):
    lines = verilog_code.strip().split("\n")
    
    # Find the last occurrence of "endmodule"
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "endmodule":
            del lines[i]
            break  # Remove only the last occurrence

    return "\n".join(lines)


with open(f'Results/Dynamic-RAG-Openai-4o-mini-Prompted-Assertion-Generation-Results-{PDF_Name}-for-New-Dataset.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Master Module','Code','golden_assertions','llm_assertions'])
    for folder in os.listdir("Evaluation/Dataset/"):
        if Excute_Folder != 'ALL_DESIGNS' and Excute_Folder not in folder:
            continue
        # folder = "arbiter"
        # if folder in ["Ripple_Carry_Adder", "or1200_operandmuxes", "gray", "Flip_Flop_Array", "PSGBusArb", "apb", "host_interface", "control_unit", "Programmable_Sequence_Detector", "PWM", "module_i2c", "delay2", "simple_req_ack", "Gray_Code_Counter", "uartTrans", "i2c", "uartRec", "APB_FSM_Controller", "register", "SEVEN","arbiter","simple_pipeline","lcd","Parallel_In_Serial_Out_Shift_Reg","fifo","or1200_if","uart_transmit","ff"]:
        #     continue
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
                    # leaf_sv_files = details
                    continue
                explanation = details.get("Assertion Explaination", "No explanation provided").lower()

                # clk_condition = "" if details.get("clock signal condition") is "none" else details.get("clock signal condition")
                # reset_condition = "" if details.get("disable condition") is "none" else details.get("disable condition")
                
                assertion_format = f"assert property (ONLY logical expression WITHOUT clock signal condition @(posedge clock) and WITHOUT disable condition disable iff(...));"
                
                keywords = extract_keywords(explanation)
                extract_operators_explanations = extract_related_operators_of_keyword(keywords)
                
                checking_str = ""
                for op_explanation in extract_operators_explanations:
                    checking_str += f"{op_explanation}\n\n"
                    retrieved_doc = code_retriever.invoke(f"{op_explanation}")                    
                    for doc in retrieved_doc:
                        checking_str += doc.page_content + "\n\n"
                    checking_str += "\n"

                prompt = f"Given Verilog code snippet as below: \n{code}\n Please generate such an assertion for it following the description:{explanation}\nThe output format should STRICTLY follow :\n{assertion_format}\nWITHOUT other things."

                llm_result = rag_chain.invoke({"keywords_explaination": checking_str, "code":code,"input":explanation,"assertion_format":assertion_format})
                llm_response = llm_result["answer"]

                completion = client.chat.completions.create(
                model= Model_Name,
                messages=[
                    {"role": "system", "content": "You are a helpful bot to extract the systemverilog assertion from the given text."},
                    {"role": "user", "content": f"Please extract the systemverilog assertion from the following text and only output its logic expression without such as `assert property` and `@(posedge clk)`:\n{llm_response}"}
                ]
                )
                logic_expression = completion.choices[0].message.content
                
                # llm_explain = rag_chain_explain.invoke({"input":logic_expression})
                # llm_response_explain = llm_explain["answer"]

                # completion = client.chat.completions.create(
                # model= Model_Name,
                # messages=[
                #     {"role": "system", "content": "You are a helpful bot to break down the given systemverilog assertion and store them in an array. Only output the array in the format: `(p1, p2, p3,...)`."},
                #     {"role": "user", "content": f"{logic_expression}\n "}
                # ]
                # )
                # property_ops = completion.choices[0].message.content

                # property_ops = str(re.search(r'\((.*?)\)',property_ops,re.DOTALL).group(1)).split(',')
                # SVA operator-based rechecking: rather than pattern-matching operators
                # present in the generated assertion against a generic glossary, parse
                # the assertion into its operator/signal syntax tree (sva_tree/sva_graph.py)
                # and compose a bottom-up, node-by-node natural-language explanation of
                # what the generated code actually means (sva_tree/explanation_merge_tree.py).
                # That derived meaning is what gets compared against the original
                # explanation below, so a mismatch points at the specific operator node
                # responsible instead of relying on a holistic re-read of raw SVA syntax.
                try:
                    merge_tree_str = build_and_render_explanation_merge_tree(
                        client, Model_Name, logic_expression, operator_context, max_retries=5
                    )
                    used_merge_tree = True
                    checking_str = (
                        "The following is a derived, node-by-node breakdown of what the "
                        f"generated assertion `{logic_expression}` actually means, built "
                        "mechanically from its parsed syntax tree. Each `Tn` line shows one "
                        "subexpression, the SVA operator that merges its operand(s) into it, "
                        f"and the resulting natural-language meaning:\n\n{merge_tree_str}"
                    )
                except ValueError:
                    # sva_graph.py couldn't parse this assertion (~15% of the corpus) --
                    # fall back to the plain operator-glossary + retrieval context.
                    used_merge_tree = False
                    checking_str = ""
                    if "|=>" in logic_expression or "|->" in logic_expression:
                        checking_str += "`|->`: \nif the left-hand side condition of |-> is true, the right-hand side condition of |-> is true in the same clock cycle\n\n\n"
                        checking_str += "`|=>`: \nif the left-hand side condition of |=> is true, the right-hand side condition of |=> is true in the next one clock cycle\n\n\n"
                    for op in _operators:
                        if op in logic_expression:
                            entry = _operators[op]
                            op_text = f"{op} ({entry['type']}): {entry['natural_langage_explanation']} Example: {entry['example_usgae']}"
                            checking_str += f"`{op_text}`\n\n"
                            retrieved_doc = code_retriever.invoke(op_text)
                            for doc in retrieved_doc:
                                checking_str += doc.page_content + "\n\n"
                            checking_str += "\n"

                recheck_instruction = (
                    "If there is a mismatch, point to the specific Tn node responsible, "
                    "list the differences, and modify it into a new systemverilog assertion "
                    "and output the new assertion.\n"
                    if used_merge_tree else
                    "If there exists a mismatch, please list the differences and modify it "
                    "into a new systemverilog assertion and output the new assertion.\n"
                )

                recheck_system_msg = (
                    "You are a helpful bot to modify the systemverilog assertion based on the given explanation.\n\n"
                    "SVA Operator Context:\n" + operator_context
                )
                completion = client.chat.completions.create(
                model= Model_Name,
                messages=[
                    {"role": "system", "content": recheck_system_msg},
                    {"role": "user", "content": f"Given the desired explanation\n{explanation},\n please check whether the systemverilog assertion {logic_expression} operates with the correct logic and the same timing (i.e., clock cycle).\n{checking_str}\n{recheck_instruction}"}
                ]
                )
                # print(completion.choices[0].message.content)
                llm_response = completion.choices[0].message.content


                # assertion checker
                nItChecker = 3
                for it in range(nItChecker):
                    checker_prompt = assertion_checker_prompt(llm_response,assertion_format)
                    llm_response = rag_chain_checker.invoke({"input":checker_prompt})["answer"]

                i += 1
                match = re.search(r'assert property\s*\(\s*(.*?)\s*\)\s*;', llm_response, re.DOTALL)
                matched_str = str(match.group(0))
                matched_str = matched_str.replace("\n"," ")
                llm_responses.append(f"\"Assertion {i}\": \"{matched_str}\"") 

            llm_response = "{\n"
            for i in range(len(llm_responses)-1):
                llm_response += llm_responses[i]+",\n"
            llm_response +=llm_responses[-1]+"\n}"
            csv_writer.writerow([folder,code,explanation_origin,llm_response])

            print(f"====================={folder} finished=====================")

            if config["JasperGold_VERIFY"] == 1:
                llm_assertions = json.loads(llm_response)
                clk_conditions = []
                disable_conditions = []
                golden_logic_expressions = []
                llm_logic_expressions = []

                for assertion, details in explanation_json.items():
                    if "Assertion" not in assertion:
                        # leaf_sv_files = details
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
                
                # combine_assertions = remove_last_endmodule(verilog_code_w_assertions)
                # combine_assertions = extract_prerequist_of_assertions(verilog_code_w_assertions,verilog_code_wo_assertions,len(clk_conditions))

                for i in range(len(clk_conditions)):
                    combine_assertion = f"assert property ({clk_conditions[i]} {disable_conditions[i]} ({llm_logic_expressions[i]}));" 
                    combine_assertions.append(combine_assertion)
                    combine_assertion = f"assert property ({clk_conditions[i]} {disable_conditions[i]} ({golden_logic_expressions[i]}) iff ({llm_logic_expressions[i]}));" 
                    combine_assertions.append(combine_assertion)
                    

                processed_code = remove_last_endmodule(verilog_code_w_assertions)
                processed_code += "\n\n"
                for assertion in combine_assertions:
                    processed_code += assertion+"\n"
                processed_code += "\nendmodule\n"

                with open(folder_path+"/"+folder+f"_KWGoldHybridDynamic-RAG-{Model_Name}.sv","w") as file:
                    file.write(processed_code)







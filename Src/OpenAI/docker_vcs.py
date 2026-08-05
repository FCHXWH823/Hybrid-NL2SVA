import os

def remove_last_endmodule(verilog_code):
    lines = verilog_code.strip().split("\n")
    
    # Find the last occurrence of "endmodule"
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "endmodule":
            del lines[i]
            break  # Remove only the last occurrence

    return "\n".join(lines)

def vcs_process(verilog_leaf_sv, verilog_code, assertion, leaf_file_name, file_name):

    processed_code = remove_last_endmodule(verilog_code)
    processed_code += "\n\n"
    processed_code += assertion+"\n"
    processed_code += "\nendmodule\n"
    # Save the processed code to a file
    with open(f"/Users/fch/Python/VC_Formal_WorkSpace/{file_name}.sv", "w") as f:
        f.write(processed_code)
    if leaf_file_name != "":
        with open(f"/Users/fch/Python/VC_Formal_WorkSpace/{leaf_file_name}.v", "w") as f:
            f.write(verilog_leaf_sv)
        command = f"docker exec recursing_shirley bash -c 'cd /mnt/VC_Formal_WorkSpace && ./vcs_2.sh {leaf_file_name} {file_name}'"
    else:
        command = f"docker exec recursing_shirley bash -c 'cd /mnt/VC_Formal_WorkSpace && ./vcs_1.sh {file_name}'"
    os.system(command)

    # vcs.sh: rm $1.rpt vcs -sverilog -assert svaext $1.sv > $1.rpt rm -rf csrc rm -rf simv.daidir rm simv rm vc_hdrs.h
    # check whether the .rpt contains "Syntax error"
    with open(f"/Users/fch/Python/VC_Formal_WorkSpace/{file_name}.rpt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Error-[" in line or "error-[" in line:
                return True
    return False

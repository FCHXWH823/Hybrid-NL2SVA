"""Verbatim port of the official CodeV-SVA prompt construction.

Source: github.com/wyt2000/CodeV-SVA
  SVAClient/src/SVAClient/Prompter.py   (get_nl2sva_human_prompt, get_nl2sva_machine_prompt)
  SVAClient/src/SVAClient/Few_Shots.py  (NL2SVA_HUMAN_EXAMPLE_*)

Copied rather than re-derived. An earlier hand-written version of this diverged
from the official one on all 356 benchmark records, in four ways:

  1. it inserted `// TODO: ASSERTION` before endmodule (a training-data
     convention the official inference path does not use) -- all 356 records
  2. it appended "Use the signals 'a', 'b'..." from signals_for_validity /
     signal_list -- 283 machine + 9 human records
  3. it emitted "Do not add code to output an error message string." only
     alongside the tb_reset line, so all 283 machine prompts lost it; the
     official code emits it unconditionally
  4. it reused the human few-shot example for machine, showing
     `disable iff (tb_reset)` to prompts whose testbench declares no tb_reset
     at all -- 283 records

(4) is the one with teeth: it demonstrates a template the model cannot satisfy.

The agents pick the variant by task: Agent_NL2SVA_Human calls the human
function with its default example_type="seq", Agent_NL2SVA_Machine calls the
machine one. Neither passes anything derived from signal_list.
"""

NL2SVA_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.\n"
    "Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.\n"
)

# --- Few_Shots.py ---
NL2SVA_HUMAN_EXAMPLE_SEQUENTIAL = '''asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);'''

NL2SVA_HUMAN_EXAMPLE_CLK_ONLY = '''asrt: assert property (@(posedge clk) 
    (a && b) != 1'b1
);'''

NL2SVA_HUMAN_EXAMPLE_COMBINATORIAL = '''asrt: assert property (
    (a && b) != 1'b1
);'''


# --- Prompter.py ---
def get_nl2sva_human_prompt(testbench: str, problem: str, example_type="seq") -> str:
    prompt = ""
    prompt += "Here is the testbench to perform your translation:\n"
    prompt += testbench
    prompt += "\nQuestion: Create a SVA assertion that checks: "
    prompt += problem + "\n"

    if example_type == "seq":
        example = NL2SVA_HUMAN_EXAMPLE_SEQUENTIAL
    elif example_type == "comb":
        example = NL2SVA_HUMAN_EXAMPLE_COMBINATORIAL
    elif example_type == "clk_only":
        example = NL2SVA_HUMAN_EXAMPLE_CLK_ONLY
    else:
        assert False, f"Invalid example_type: {example_type}!"

    if example_type == "seq":
        prompt += "You should use `tb_reset` as the disable condition signal. "
    prompt += f"""Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
{example}
```
Answer:"""
    return prompt


def get_nl2sva_machine_prompt(problem: str, testbench: str) -> str:
    prompt = ""
    prompt += "Here is the testbench to perform your translation:\n"
    prompt += testbench
    prompt += "\nQuestion: Create a SVA assertion that checks: "
    prompt += problem + "\n"
    prompt += """Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
assert property (@(posedge clk)
    (sig_A && sig_B) != 1'b1
);
```
Answer:"""
    return prompt


def build_user_prompt(record, task):
    """Dispatch by task, mirroring Agent_NL2SVA_Human / Agent_NL2SVA_Machine."""
    if task == "human":
        return get_nl2sva_human_prompt(testbench=record["testbench"],
                                       problem=record["problem"])
    if task == "machine":
        return get_nl2sva_machine_prompt(problem=record["problem"],
                                         testbench=record["testbench"])
    raise ValueError("task must be 'human' or 'machine', got %r" % task)

"""
Stage 6 of the CodeV-SVA OL-NL + two-part decomposition pipeline (see the plan
in /Users/fch/.claude/plans/polished-chasing-wreath.md).

Runs the OL-NL/DFS-fine-tuned model (Stage 5's output, served behind any
OpenAI-compatible endpoint -- e.g. vLLM's `--served-model-name`) over one of
the three benchmark CSVs and writes an LMResult-shaped CSV that the existing
FVEval harness (Evaluation/FVRuleLearner/FVEval/fv_eval/evaluation.py --
NL2SVAHumanEvaluator / NL2SVAMachineEvaluator) can consume as-is, via its
`llm_output_dir` glob of `*{model_name}_*.csv`. No changes are made to that
harness; this script only produces its expected input.

Prompt construction deliberately does NOT force the training data's "use
tb_reset as the disable condition" instruction: spot-checking the benchmark
CSVs' own ref_solution column shows that convention isn't followed
consistently even within a single benchmark (e.g. nl2sva_machine task 3_0_0's
reference has no `disable iff` at all despite its testbench defining
tb_reset), so hard-coding that instruction at eval time would actively bias
the model against some of the benchmark's own reference answers. Instead each
row gets the same *shape* of prompt used in training (testbench-with-
`// TODO: ASSERTION` + "Question: <prompt>" + output-format instructions +
worked example), letting the model decide whether tb_reset applies the same
way it must at real inference time.

`module_sva_nl_manual_editing.csv` only provides `module_interface` (a bare
port list, no body) -- it's wrapped into a minimal module with the injection
marker appended directly, since there's no tb_reset scaffold to preserve.

CAVEAT: the exact fields the harness's property-equivalence check consumes
beyond `response`/`ref_solution` (specifically `output_tb`, used by
write_testbench_sv/launch_jg_custom_equiv_check) were not fully traced against
a live JasperGold run in this environment -- `output_tb` is populated with the
same testbench text as `design_rtl` here as a reasonable default; confirm
against fv_tool_execution.py once JasperGold is actually available.

Usage:
    python verilogFinetune/run_codev_sva_ol_dfs_eval.py \\
        --csv Evaluation/FVRuleLearner/FVEval/data_nl2sva/data/nl2sva_human.csv \\
        --task nl2sva_human \\
        --endpoint-base-url http://localhost:8000/v1 \\
        --model deepseek-coder-7b-codev-sva-ol-dfs \\
        --output verilogFinetune/eval_outputs/nl2sva_human_deepseek-coder-7b-codev-sva-ol-dfs.csv
"""
import argparse
import csv
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = (
    "You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.\n"
    "Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.\n"
)

QUESTION_TEMPLATE = """Question: Create a SVA assertion that checks: {prompt}
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Answer:"""


def build_testbench_with_marker(raw_testbench):
    """Inserts the `// TODO: ASSERTION` injection point right before the
    final `endmodule`, matching the training data's convention."""
    if "endmodule" in raw_testbench:
        head, _, _ = raw_testbench.rpartition("endmodule")
        return f"{head.rstrip()}\n// TODO: ASSERTION\nendmodule"
    return f"{raw_testbench.rstrip()}\n// TODO: ASSERTION\nendmodule"


def build_user_prompt(testbench_with_marker, prompt_text):
    return (
        "Here is the testbench to perform your translation:\n"
        f"{testbench_with_marker}\n"
        f"{QUESTION_TEMPLATE.format(prompt=prompt_text)}"
    )


def iter_rows(csv_path, task):
    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader):
            if task == "module_sva_nl_manual_editing":
                raw_testbench = f"{row['module_interface'].rstrip()}"
                task_id = f"{row['design_name']}_{row_number}"
            else:
                raw_testbench = row["testbench"]
                task_id = row["task_id"]
            yield task_id, raw_testbench, row["prompt"], row["ref_solution"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to one of the 3 benchmark CSVs")
    parser.add_argument("--task", required=True, choices=["nl2sva_human", "nl2sva_machine", "module_sva_nl_manual_editing"])
    parser.add_argument("--endpoint-base-url", required=True, help="OpenAI-compatible base URL (e.g. a vLLM server)")
    parser.add_argument("--api-key", default="EMPTY", help="Most local OpenAI-compatible servers ignore this")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    client = OpenAI(base_url=args.endpoint_base_url, api_key=args.api_key)
    experiment_id = os.path.basename(args.csv).split(".csv")[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fieldnames = [
        "experiment_id", "task_id", "model_name", "response", "ref_solution",
        "design_rtl", "output_tb", "user_prompt", "cot_response",
    ]
    with open(args.output, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, (task_id, raw_testbench, prompt_text, ref_solution) in enumerate(iter_rows(args.csv, args.task)):
            if args.limit is not None and row_index >= args.limit:
                break

            testbench_with_marker = build_testbench_with_marker(raw_testbench)
            user_prompt = build_user_prompt(testbench_with_marker, prompt_text)

            print(f"[{row_index + 1}] task_id={task_id} ...")
            last_error = None
            response_text = None
            for attempt in range(args.max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    response_text = completion.choices[0].message.content
                    break
                except Exception as error:
                    last_error = error
                    print(f"    call failed ({error}), retrying...")
            if response_text is None:
                raise RuntimeError(f"Failed on task_id={task_id} after {args.max_retries} attempts") from last_error

            writer.writerow({
                "experiment_id": experiment_id,
                "task_id": task_id,
                "model_name": args.model,
                "response": response_text,
                "ref_solution": ref_solution,
                "design_rtl": testbench_with_marker,
                "output_tb": testbench_with_marker,
                "user_prompt": user_prompt,
                "cot_response": "",
            })
            out_file.flush()

    print(f"Wrote responses to {args.output}")


if __name__ == "__main__":
    main()

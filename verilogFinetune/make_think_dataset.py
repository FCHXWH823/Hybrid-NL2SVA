"""Rewrap the OL-NL/DFS reasoning dataset into reasoning-model (<think>) format.

`codev_sva_ol_dfs_5000.jsonl`'s assistant turn is

    OL NL: ...
    ***nl-decomposition tree***
    ...
    ***operator-merge-sva tree***
    ...
    Final assertion (Tn): ...

    ```systemverilog
    <golden SVA>
    ```

i.e. the reasoning and the answer sit side by side as plain text. Training a
reasoning model wants that split explicitly: everything up to the final
` ```systemverilog ` fence is the chain of thought and belongs inside
`<think>...</think>`; the fenced block itself stays outside as the answer.

The split point is the *last* ` ```systemverilog ` fence -- the same anchor the
eval harness's `utils.parse_code_response` uses -- so the emitted answer is
byte-for-byte the block that scoring will extract. System and user turns are
copied through unchanged.
"""

import argparse
import json
import os

FENCE = "```systemverilog"


def split_assistant(content):
    """-> (reasoning, code_block). Raises ValueError if the turn isn't shaped
    the way every record in the source dataset is (reasoning, then exactly one
    trailing fenced SVA block)."""
    idx = content.rfind(FENCE)
    if idx == -1:
        raise ValueError("no %s fence" % FENCE)
    reasoning = content[:idx].strip()
    code_block = content[idx:].strip()
    if not reasoning:
        raise ValueError("empty reasoning before fence")
    if not code_block.endswith("```"):
        raise ValueError("fenced block not closed")
    return reasoning, code_block


def to_think_format(content, open_tag, close_tag):
    reasoning, code_block = split_assistant(content)
    return "%s\n%s\n%s\n\n%s" % (open_tag, reasoning, close_tag, code_block)


def convert(in_path, out_path, open_tag, close_tag):
    total = written = 0
    failures = []
    with open(in_path) as fin, open(out_path, "w") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            messages = record["messages"]
            try:
                out_messages = [
                    dict(m, content=to_think_format(m["content"], open_tag, close_tag))
                    if m["role"] == "assistant"
                    else m
                    for m in messages
                ]
            except ValueError as exc:
                failures.append((lineno, str(exc)))
                continue
            fout.write(json.dumps(dict(record, messages=out_messages), ensure_ascii=False) + "\n")
            written += 1
    return total, written, failures


def register(dataset_info_path, name, file_name):
    """Add a sharegpt entry mirroring codev_sva_ol_dfs_5000's own registration."""
    with open(dataset_info_path) as f:
        info = json.load(f)
    info[name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    # indent=4 matches the file's existing style; anything else silently
    # reformats all ~90 lines and buries the one-entry addition in the diff.
    with open(dataset_info_path, "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(here, "data", "codev_sva_ol_dfs_5000.jsonl"))
    parser.add_argument("--output", default=os.path.join(here, "data", "codev_sva_ol_dfs_5000_think.jsonl"))
    parser.add_argument("--open-tag", default="<think>")
    parser.add_argument("--close-tag", default="</think>")
    parser.add_argument("--dataset-name", default="codev_sva_ol_dfs_5000_think",
                        help="name to register in dataset_info.json")
    parser.add_argument("--no-register", action="store_true",
                        help="skip updating data/dataset_info.json")
    args = parser.parse_args()

    total, written, failures = convert(args.input, args.output, args.open_tag, args.close_tag)
    print("read %d records, wrote %d -> %s" % (total, written, args.output))
    if failures:
        print("SKIPPED %d malformed record(s):" % len(failures))
        for lineno, why in failures[:10]:
            print("  line %d: %s" % (lineno, why))

    if not args.no_register:
        info_path = os.path.join(os.path.dirname(args.output), "dataset_info.json")
        register(info_path, args.dataset_name, os.path.basename(args.output))
        print("registered '%s' in %s" % (args.dataset_name, info_path))


if __name__ == "__main__":
    main()

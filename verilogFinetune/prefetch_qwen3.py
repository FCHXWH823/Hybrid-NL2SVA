"""Pre-download the Qwen3 base models into HF_HOME.

train_qwen3_think.sh passes a bare repo id (Qwen/Qwen3-8B) as
--model_name_or_path, so without this the weights are fetched on first use --
inside the H200 allocation, spending GPU-hours on network I/O and taking the
whole job down if the transfer drops. Running it here, on a CPU node, makes
the training job's model load a local cache hit.

Verifies each download by loading the config and walking the safetensors index
to confirm every shard named by the index is actually present on disk -- a
truncated download otherwise surfaces much later, as a confusing load error
inside the training job.
"""

import argparse
import json
import os
import sys

DEFAULT_MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]

# Skip alternate weight formats some repos ship alongside safetensors; keeping
# them would double the download for no benefit.
IGNORE = ["original/*", "*.pth", "*.gguf", "*.msgpack", "*.h5"]


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return "%.1f%s" % (n, unit)
        n /= 1024
    return "%.1fPB" % n


def verify(path, repo_id):
    """Confirm the snapshot is complete, not just present."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False, "config.json missing"
    with open(cfg_path) as f:
        cfg = json.load(f)

    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            shards = set(json.load(f)["weight_map"].values())
        missing = [s for s in sorted(shards) if not os.path.isfile(os.path.join(path, s))]
        if missing:
            return False, "%d shard(s) missing, e.g. %s" % (len(missing), missing[0])
        detail = "%d shards" % len(shards)
    elif os.path.isfile(os.path.join(path, "model.safetensors")):
        detail = "single-file weights"
    else:
        return False, "no safetensors weights found"

    tok = any(
        os.path.isfile(os.path.join(path, n))
        for n in ("tokenizer.json", "tokenizer_config.json")
    )
    if not tok:
        return False, "tokenizer files missing"

    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
            else:
                real = os.path.realpath(fp)
                if os.path.isfile(real):
                    total += os.path.getsize(real)

    return True, "%s, %s, arch=%s" % (detail, human(total), cfg.get("architectures", ["?"])[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print("HF_HOME =", os.environ.get("HF_HOME", "(unset)"))
    print()

    failures = []
    for repo_id in args.models:
        print("=" * 70)
        print("downloading", repo_id)
        print("=" * 70)
        try:
            path = snapshot_download(
                repo_id=repo_id,
                ignore_patterns=IGNORE,
                max_workers=args.max_workers,
            )
        except Exception as exc:
            print("FAILED: %s: %s" % (type(exc).__name__, exc))
            failures.append(repo_id)
            continue

        ok, detail = verify(path, repo_id)
        print("path  :", path)
        print("verify:", "OK" if ok else "INCOMPLETE", "--", detail)
        if not ok:
            failures.append(repo_id)
        print()

    print("=" * 70)
    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)
    print("ALL MODELS DOWNLOADED AND VERIFIED")


if __name__ == "__main__":
    main()

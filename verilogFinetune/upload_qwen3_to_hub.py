"""Upload a fine-tuned Qwen3 checkpoint to the Hugging Face Hub.

Follows the convention of the existing save_*_to_hub.py scripts -- the Jalik/
namespace, private repos, token from $HUGGINGFACE_TOKEN -- but uploads the
directory directly instead of round-tripping through
AutoModelForCausalLM.from_pretrained(...).push_to_hub(...).

Two reasons for the difference:

  * Memory. Loading an 8B model to re-serialize it needs ~16GB of RAM (or a
    GPU, via the old scripts' device_map='auto'). upload_folder streams file
    by file, so this runs anywhere -- including a login node under a 3GiB cap.
  * Fidelity. The trainer already wrote correct safetensors shards, an index,
    the tokenizer, and the generation config. Re-serializing risks subtle
    drift (dtype, shard boundaries, missing added_tokens.json); uploading the
    directory publishes exactly the bytes that were evaluated.

Deliberately excludes checkpoint-*/ (intermediate training state, not part of
the model) and optimizer/RNG state, which are large and useless to a consumer.
"""

import argparse
import json
import os
import sys

DEFAULT_LOCAL = "/scratch/wx2356/verilogFinetune/output/qwen3-8B-codev-sva-ol-dfs-think"
DEFAULT_REPO = "Jalik/qwen3-8b-codev-sva-ol-dfs-think"

# Training-only artifacts: not needed to load or serve the model.
IGNORE = [
    "checkpoint-*",
    "*.pth",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
    "zero_to_fp32.py",
]


def resolve_token():
    """Env var first, Src/Config.yml second -- the same order the generation
    scripts use for their API keys, so a key never has to be exported by hand
    to be picked up. Config.yml is gitignored, so the token stays out of git."""
    for var in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]

    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Src", "Config.yml")
    if os.path.isfile(cfg):
        try:
            import yaml
            with open(cfg) as f:
                data = yaml.safe_load(f) or {}
            for key in ("HUGGINGFACE_TOKEN", "HuggingFace_Token", "HF_TOKEN"):
                if data.get(key):
                    return str(data[key]).strip()
        except Exception as exc:
            print("warning: could not read %s: %s" % (cfg, exc))
    return None


def preflight(local_dir):
    """Fail before touching the network if the directory isn't a complete model."""
    problems = []
    for required in ("config.json", "tokenizer_config.json"):
        if not os.path.isfile(os.path.join(local_dir, required)):
            problems.append("missing %s" % required)

    index = os.path.join(local_dir, "model.safetensors.index.json")
    single = os.path.join(local_dir, "model.safetensors")
    if os.path.isfile(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        missing = [s for s in shards if not os.path.isfile(os.path.join(local_dir, s))]
        if missing:
            problems.append("%d shard(s) missing (e.g. %s)" % (len(missing), missing[0]))
        else:
            print("weights: %d shards, all present" % len(shards))
    elif os.path.isfile(single):
        print("weights: single-file safetensors")
    else:
        problems.append("no safetensors weights found")

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--private", dest="private", action="store_true", default=True,
                        help="create the repo private (default, matches save_*_to_hub.py)")
    parser.add_argument("--public", dest="private", action="store_false",
                        help="create the repo PUBLIC -- anyone can download it")
    parser.add_argument("--dry-run", action="store_true",
                        help="run preflight and list what would upload, then stop")
    args = parser.parse_args()

    token = resolve_token()
    if not token and not args.dry_run:
        sys.exit(
            "ERROR: no token. Set HUGGINGFACE_TOKEN (a WRITE token from\n"
            "https://huggingface.co/settings/tokens), or add HUGGINGFACE_TOKEN\n"
            "to Src/Config.yml, or run `huggingface-cli login`."
        )

    if not os.path.isdir(args.local_dir):
        sys.exit("ERROR: not a directory: %s" % args.local_dir)

    problems = preflight(args.local_dir)
    if problems:
        sys.exit("ERROR: %s is not a complete model:\n  - %s"
                 % (args.local_dir, "\n  - ".join(problems)))

    total = 0
    names = []
    for entry in sorted(os.listdir(args.local_dir)):
        path = os.path.join(args.local_dir, entry)
        if os.path.isfile(path):
            total += os.path.getsize(path)
            names.append(entry)
    print("source    : %s" % args.local_dir)
    print("repo      : %s (%s)" % (args.repo_id, "private" if args.private else "PUBLIC"))
    print("uploading : %d files, %.1f GB" % (len(names), total / 1024 ** 3))
    print("excluding : %s" % ", ".join(IGNORE))

    if args.dry_run:
        print("\n[dry run] nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private,
                    repo_type="model", exist_ok=True)

    # create_repo(exist_ok=True) does NOT change the visibility of a repo that
    # already exists, so an earlier private attempt would otherwise silently
    # stay private -- and a private push fails once the account's private
    # storage quota is reached. Reconcile explicitly.
    info = api.repo_info(repo_id=args.repo_id, repo_type="model")
    if bool(info.private) != bool(args.private):
        print("visibility: %s -> %s"
              % ("private" if info.private else "public",
                 "private" if args.private else "public"))
        api.update_repo_settings(repo_id=args.repo_id, repo_type="model",
                                 private=args.private)
    else:
        print("visibility: already %s" % ("private" if args.private else "public"))
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=IGNORE,
        commit_message="Add Qwen3-8B fine-tuned on codev_sva_ol_dfs_5000_think",
    )
    print("\nDONE -> https://huggingface.co/%s" % args.repo_id)


if __name__ == "__main__":
    main()

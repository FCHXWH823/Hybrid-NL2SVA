# Hybrid-NL2SVA

Research code for automatically generating SystemVerilog Assertions (SVAs) from natural-language
property descriptions and RTL designs (the *NL2SVA* task), combining a customized retrieval-augmented
generation (RAG) pipeline with LLM fine-tuning on reasoning-annotated datasets.

Manually writing SVAs requires translating a natural-language requirement into formally correct,
signal-grounded assertion syntax — a task where general-purpose LLMs are unreliable without either
relevant retrieved context (operator semantics, similar assertions, design-specific signal
information) or targeted fine-tuning on how an SVA's structure is actually derived from its
description. This repo explores both directions and evaluates them against several NL2SVA benchmarks.

## Repository structure

- **`Src/`** — RAG-based assertion generation pipelines (static and dynamic retrieval, prompted and
  unprompted, multi-round prompting, query expansion), across OpenAI, CodeLlama, DeepSeek, and other
  model backends. `Src/Config.yml` holds API keys and run configuration (kept blank in git — fill in
  your own keys locally).
- **`verilogFinetune/`** — dataset generation and fine-tuning pipeline for teaching a model to *reason*
  its way to an SVA rather than free-associate one: derives an operator-level natural-language (OL NL)
  restatement of a spec, decomposes it top-down per-operator, and re-derives the assertion bottom-up as
  a symbolic proof, validated by formal equivalence checking (JasperGold) against the golden SVA.
  See `verilogFinetune/framework_context/OL_NL_DFS_FRAMEWORK.md` for the full pipeline writeup.
- **`Evaluation/`** — benchmark datasets and evaluation harnesses, including `FVRuleLearner/` (a vendored
  copy of the FVEval benchmark/scoring harness — nested git repo, not tracked here; see
  `Evaluation/FVRuleLearner`'s own README) and hand-curated assertion/evaluation datasets.
- **`Results/`** — generated result CSVs from running the various pipelines against the evaluation
  datasets.
- **`PlotFigures/`** — scripts and output figures summarizing experiment results.
- **`RAG_Database/`, `SVTextbooks/`, `VerilogTextBooks/`** — source corpora used for retrieval (SVA/
  Verilog textbooks and reference material).
- **`RelatedWorks/`** — reference papers for related NL2SVA/assertion-mining approaches.
- **`TCAD-Hybrid-NL2SVA/`** — LaTeX source for the paper this repo accompanies (nested git repo, not
  tracked here).
- **`operators.json`, `sva_property_sequence_operators.jsonl`** — SVA operator reference tables used
  throughout the generation/explanation prompts.

## Setup

```
pip install -r requirement.txt
```

Fill in your own API keys in `Src/Config.yml` (`Openai_API_Key`, `DeepSeek_API_Key`, etc. — left blank
in version control).

## Where to start

- For the RAG-based generation pipelines: see the scripts under `Src/` (e.g.
  `Src/RAG-Openai-4o-mini-Prompted-Assertion-Generation-1assert1iteration.py`).
- For the fine-tuning dataset pipeline: see `verilogFinetune/framework_context/OL_NL_DFS_FRAMEWORK.md`
  and `verilogFinetune/generate_codev_sva_reasoning_dataset.py`.
- For benchmark evaluation: see `Evaluation/FVRuleLearner/FVEval/`.

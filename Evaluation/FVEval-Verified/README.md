---
license: apache-2.0
---

# CodeV-SVA: Training Specialized LLMs for Hardware Assertion Generation via RTL-Grounded Bidirectional Data Synthesis

<div align="center"> 
  <a href="https://huggingface.co/wyt2000/CodeV-SVA-14B"><img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://github.com/wyt2000/CodeV-SVA"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a> &ensp;
</div>

## Introduction

We introduce CodeV-SVA, a family of large language models designed to translate natural-language verification properties into SystemVerilog Assertions (SVAs).

Open-Source Plan:

- Model ✓
- Evaluation code ✓
- Paper
- Dataset
- Data synthesis and training code

We employ human experts to recheck [FVEval](https://github.com/NVlabs/FVEval) benchmark, correcting or removing erroneous tests (see `fveval_nl2sva_human.jsonl` and `fveval_nl2sva_machine.jsonl`). 

we will provide a detailed description of the modifications and enhancements made to these two benchmarks in future.

## Citation

```latex
@misc{CodeV-SVA,
    title={CodeV-SVA: Training Specialized LLMs for Hardware Assertion Generation via RTL-Grounded Bidirectional Data Synthesis}, 
    author={Yutong Wu and Chenrui Cao and Pengwei Jin and Di Huang and Rui Zhang and Xishan Zhang and Zidong Du and Qi Guo and Xing Hu},
    year={2025},
    howpublished={\url{https://huggingface.co/wyt2000/CodeV-SVA-14B}},
}
```
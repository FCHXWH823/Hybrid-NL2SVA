# AssertionForge

AssertionForge is a research prototype for **generating high‑quality SystemVerilog Assertions (SVAs)** from natural‑language specifications and RTL.  It constructs a joint knowledge graph that bridges the semantic gap between spec and implementation, then leverages LLMs to produce a focused test plan and candidate SVAs.

## Project Overview

AssertionForge enhances formal verification assertion generation with structured representation of specifications and RTL. The project follows a two-stage workflow:

1. **Knowledge Graph Construction (Indexing)**
2. **Test Plan and SVA Generation**

## Setup and Usage

Before running any command, always activate the virtual environment:

```bash
cd /<path>/<to>/src && conda activate fv
```

## Working with a New Design

For a new design, you'll need to set specific parameters in config.py for both stages. Here's what you need to modify:

### Common Parameters for New Designs

- `design_name`: A unique identifier for your design (e.g., 'my_new_design')
- Create appropriate paths for your design's files:
  - Specification document (PDF)
  - RTL code directory (containing .v files)
  - Output directory for the Knowledge Graph

## Stage 1: Knowledge Graph Construction (Indexing)

1. Edit `/<path>/<to>/src/config.py`:
   - Set `task = 'build_KG'`
   - Set `design_name` to your new design name
   - Set paths for your design:
     ```python
     input_file_path = "/path/to/your/specification.pdf"
     ```
   - Keep GraphRAG paths as standard (usually don't need to change):
     ```python
     env_source_path = "/<path>/<to>/rag_apb/.env"
     settings_source_path = "/<path>/<to>/rag_apb/settings.yaml"
     entity_extraction_prompt_source_path = "/<path>/<to>/rag_apb/prompts/entity_extraction.txt"
     graphrag_local_dir = "/<path>/<to>/graphrag"
     ```

2. Run the indexing:
   ```bash
   python main.py
   ```

3. **Note the KG output path from the console** - you'll need it for Stage 2. It will be something like:
   ```
   /<path>/<to>/data/your_design_name/spec/graph_rag_your_design_name/output/[timestamp]/artifacts/clustered_graph.0.graphml
   ```

## Stage 2: Test Plan and SVA Generation

1. Edit `/<path>/<to>/src/config.py`:
   - Set `task = 'gen_plan'`
   - Set `subtask = 'actual_gen'`
   - Configure design parameters:
     ```python
     design_name = "your_design_name"  # Same as in Stage 1
     file_path = "/path/to/your/specification.pdf"  # Same as input_file_path from Stage 1
     design_dir = "/path/to/your/rtl/directory"  # Directory containing your design's .v files
     KG_path = "/path/from/stage1/output/clustered_graph.0.graphml"  # Path noted from Stage 1
     ```
   - Set architectural signals:
     ```python
     gen_plan_sva_using_valid_signals = True
     valid_signals = ['signal1', 'signal2']  # Replace with your design's actual signal names
     ```
   - For new designs, disable SVA generation:
     ```python
     generate_SVAs = False  # Important for designs without TCL files
     ```
   - LLM configuration (usually keep as is):
     ```python
     llm_model = "gpt-4o"
     use_KG = True
     prompt_builder = "dynamic"
     ```

2. Run the test plan generation:
   ```bash
   python main.py
   ```

## Parameter Details for New Designs

### Required Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `design_name` | Unique name for your design | `"my_custom_asic"` |
| `input_file_path` / `file_path` | Path to specification PDF | `"/home/user/specs/my_design_spec.pdf"` |
| `design_dir` | Directory containing RTL (.v) files | `"/home/user/rtl/my_design/"` |
| `KG_path` | Path to KG from Stage 1 | Output path from Stage 1 |
| `valid_signals` | List of architectural signals | `['clk', 'reset', 'data_valid']` |

### Optional Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DEBUG` | Enable faster processing (fewer signals) | `False` |
| `max_num_signals_process` | Limit number of signals to process | `float('inf')` |
| `max_prompts_per_signal` | Number of prompts per signal | `3` |
| `generate_SVAs` | Whether to generate SVA code | `False` |

## Important Notes

- The KG construction process may take some time depending on the size of the specification.
- Always check the console output for any errors or warnings during the process.
- For new designs, keep `generate_SVAs = False` since TCL files might not be provided.
- Always specify `valid_signals` with the actual architectural signals from your design.
- Architectural signals are typically input/output ports and architectural-level registers mentioned in the specification.

## Example Workflow for a New Design

```bash
# Activate environment
cd /<path>/<to>/src && conda activate fv

# Edit config.py for build_KG task with your design information
# Then run:
python main.py

# When KG construction is complete, note the output path
# Edit config.py for gen_plan task with the correct KG_path
# Then run:
python main.py
```

This is the recommended workflow for reliable operation of AssertionForge with new designs.


## Knowledge Graph Example

![OpenMSP430 Knowledge Graph](src/KG_vis.png)

*Visualization of the KGs from **OpenMSP430** using Gephi. Node colors represent different entity types—modules, signals, etc.  Left: KG generated with the vanilla GraphRAG prompt.  Middle: KG produced by our domain‑customized prompt.  Right: two zoomed‑in views highlighting key entities and their labels.*



## Citation

If you build on AssertionForge, please cite our LAD 2025 paper:

```
@inproceedings{bai2025assertionforge,
  title={AssertionForge: Enhancing Formal Verification Assertion Generation with Structured Representation of Specifications and RTL},
  author={Bai, Yunsheng and Bany Hamad, Ghaith and Suhaib, Syed and Ren, Haoxing},
  booktitle={Proceedings of the IEEE International Conference on LLM-Aided Design (LAD)},
  address={Stanford, CA},
  year={2025}
}
```

*Accepted at LAD 2025, Stanford (June 26‑27, 2025).*

📄 **Paper:** [arXiv:2503.19174](https://arxiv.org/abs/2503.19174)


##  Acknowledgements

We deeply thank Vigyan Singhal for his technical guidance and support. We also acknowledge Cadence Design Systems for implementing the formal assertion-to-assertion equivalence checking in Jasper and for their many helpful discussions that contributed to the success of this project.


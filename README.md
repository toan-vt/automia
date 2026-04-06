<h1 align="center"> Differentially Private Synthetic Data via Foundation Model APIs 2: Text</h1>

<p align="center">
<a href="https://arxiv.org/abs/2603.19375">📃 Paper</a>
•
<a href="http://toan-vt.github.io/automia" >Demo</a>
</p>

## Installation

To install `automia`(editable):
> python -m pip install -e ".[all]"

or `automia`(package)
> pip install automia

A sandbox environment (`env.sh`) that will be used to execute the generated MIAs should be installed with common computation packages (such as numpy, torch, scipy,..) and packages required to run the MIAs. We recommend to save the model's outputs (i.e., logits) using safetensor and read load to save computational cost each trial.

## Running AutoMIA for a MIA setting:

1. Prepare a codebase template including `env.sh`, `template.py`, and `config.yaml`. Example in (`examples/bbllm/arxiv_pythia`)

2. Run vllm servers. `automia` requires two APIs (LLM and embedding) and then run automia
> python -m automia.main --experiment-dir examples/bbllm/arxiv_pythia --timeout 300 --model-name qwen --base-url http://localhost:9800/v1 --provider vllm --embedding-model-name qwen-embedding --embedding-base-url http://localhost:9700/v1  --budget 100 --output-dir results/bbllm/arxiv

3. `eval.py` and `vis.py` are available to eval the top 10 MIAs and visualize the MIAs

> python vis.py --output-dir results/bbllm/arxiv

A html file index.html will be written into results/bbllm/arxiv

> python eval.py --template examples/bbllm/arxiv_pythia/template.py --output-dir results/bbllm/arxiv

A cvs file will be written into results/bbllm/arxiv

```
Please make sure the stored data is available to reload for our example template.py (line 135 in examples/bbllm/arxiv_pythia/template.py) by running generate.py scripts.
The template.py must have an argument of `output-dir`, create this output directory, and write the results into `mia-results.json` with 4 keys: "auc_score", "tpr_1_score", "tpr_5_score", "combined_score".
A detailed documents and step-by-step to reproceduce our paper's experiments will be available soon! Feel free to reach out to me at <email> or create issues.
``` 

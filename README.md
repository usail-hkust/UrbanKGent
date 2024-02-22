# UrbanKGent: A Unified Large Language Model Agent Framework for Urban Knowledge Graph Construction

<p align="center">

![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
</p>

<p align="center">

| **[1 Introduction](#introduction)** 
| **[2 Requirements](#requirements)**
| **[3 Usage](#usage)**
| **[4 Citation](#citation)**
| **[Website](https://htmlpreview.github.io/?https://raw.githubusercontent.com/usail-hkust/UrbanKGent/main/UrbanKGent%20Demo/index.html)** |

</p>

<a id="introduction"></a>
## 1 Introduction

Official code for article "[UrbanKGent: A Unified Large Language Model Agent Framework for Urban Knowledge Graph Construction](https://arxiv.org/pdf/2402.06861.pdf)".

Urban knowledge graph has recently worked as an emerging building block to distill critical knowledge from multi-sourced urban data for diverse urban application scenarios. Despite its promising benefits, urban knowledge graph construction (UrbanKGC) still heavily relies on manual effort, hindering its potential advancement. This paper presents UrbanKGent, a unified large language model agent framework, for urban knowledge graph construction.

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/usail-hkust/UrbanKGent/blob/main/UrbanKGent%20Demo/img/model.png" width="400">
</div>


<a id="requirements"></a>
## 2 Requirements

`python=3.8`,`torch=2.1.1`, `transformers=4.35.2`, `accelerate=0.25.0`, `geographiclib=2.0`, `request=2.31.0`,  `geopy=2.4.1`, `geohash2=1.1`

<a id="usage"></a>

## 3 Usage

### Agent Inference
To run OpenAI LLM agent, you need to set your key in `main_TE/KGC.py`:

```
python main_KGC/TE.py
```

To run the official Llama-2 LLM agent, you should put your LLMs model into the '/data/llm_models', and then open the Llama-2 inference local-host request: 

```
python utils/open_llm_api.py
python main_KGC/TE.py
```
To run the fine-tuned Llama-2 LLM agent, you should put your LLMs model into the '/data/llm_models', and put your LLM adapter into the '/data/llm_model/', and then open the finetuned Llama-2 inference local-host request: 
```
python utils/llama_lora_api.py
python main_KGC/TE.py
```

### Agent Fine-tuning
To fine-tune tailored LLM agent, you need to prepare instruction set and put them into './sft_data/', and then finetune the model with LoRA architecture:
```
python finetune.py
```

Our raw data, fine-tuned LLM adapter are available at "[Google Cloud](https://drive.google.com/drive/folders/1OLK1_8qN_1hNDaBzxPoTkYP5ppIfWXVI?usp=sharing)"

### Agent Downloading[![Testing Status](https://github.com/usail-hkust/UrbanKGent/blob/main/UrbanKGent%20Demo/img/huggingface_UrbanKGent.png)](https://huggingface.co/USAIL-HKUSTGZ/UrbanKGent-13b)

The opensource UrbanKGent-13b in this work could be downloaded from hugging face:
```
from huggingface_hub import snapshot_download

repo_id = "USAIL-HKUSTGZ/UrbanKGent-13b"  # 
local_dir = "./data/llm_models/UrbanKGent-13b"  # 
local_dir_use_symlinks = False  #
token = "YOUR_KEY"  # hugging face access token

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=local_dir_use_symlinks,
    token=token
)
```

## 4 Citation

If you find our work is useful for your research, please consider citing:

```
@article{ning2024urbankgent,
  title={UrbanKGent: A Unified Large Language Model Agent Framework for Urban Knowledge Graph Construction},
  author={Ning, Yansong and Liu, Hao},
  journal={arXiv preprint arXiv:2402.06861},
  year={2024}
}
```

```
@article{ning2024uukg,
  title={UUKG: unified urban knowledge graph dataset for urban spatiotemporal prediction},
  author={Ning, Yansong and Liu, Hao and Wang, Hao and Zeng, Zhenyu and Xiong, Hui},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

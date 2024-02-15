# RAP for RelationPrompt

The original code of RelationPrompt is [here](https://github.com/declare-lab/RelationPrompt). And for our experiments, we just simply implement the Extractor part of RelationPrompt.


## Requirements

General

- Python (verified on 3.7)

Python Packages

- see requirements.txt

```bash
conda create -n relationprompt python=3.7
conda activate relationprompt
pip install -r requirements.txt
```

## Split Data
Run the following commands to randomly split the datasets.

```bash
cd outputs/data
bash run_sample.bash
```

For each datasets, we get the corresponding low resource  folders `nyt_retrieved_ratio` and `webnlg_retrieved_ratio`  with 5 different random seeds, taking `nyt_retrieved_ratio` as example, we can get folders as follow:

```text
data/nyt_retrieved_ratio
├── seed2
│   ├── 0.1
│   ├── 0.01
│   └── 0.05
├── seed3
│   └── ...
├── seed5
│   └── ...
├── seed7
│   └── ...
└── seed11
    └── ...
```

## Usage

**1. Train**

Run the following command for training, `seed` here indicate the different random seed used when splitting datasets. 

```bash
python train.py --seed 2 --ratio 0.01 --data_name webnlg_retrieved -lr 5e-4 --eval_steps 20
```

**2. Evaluate**

Run the following command for evaluation (The evaluation is also conducted after the training)  :

```bash
python predict.py --seed 2 --ratio 0.01 --data_name webnlg_retrieved -lr 5e-4
```

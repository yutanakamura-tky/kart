# KART: Privacy Leakage Framework of Language Models Pre-trained with Clinical Records
This is an implementation of our arXiv preprint paper (https://arxiv.org/abs/2101.00036) "KART: Privacy Leakage Framework of Language Models Pre-trained with Clinical Records."  

## Usage
### 0. Requirements

- Python 3.6.4
- Make sure that `$HOME` is set to environment variable `$PYTHONPATH`.

### 1. Preparation
#### 1-1. Get Poetry

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py > ~/get-poetry.py
cd ~
python get-poetry.py --version  1.1.4
```

```sh
poetry config virtualenvs.in-project true
```

#### 1-2. Clone Repository & Install Packages

```sh
cd ~
git clone git@github.com:yutanakamura-tky/kart.git
cd ~/kart
poetry install
```

#### 1-3. Make MIMIC-III-dummy-PHI
```
cd ~/kart/src
bash make_mimic_iii_dummy_phi.sh
```

#### 1-4. Get non-domain-specific uncased BERT-base model
```
cd ~/kart/src
bash get_google_bert_model.sh
```

#### 1-5. Convert MIMIC-III to BERT pre-training data 
```
cd ~/kart/src
bash make_pretraining_data.sh
```

#### 1-6. Pre-train BERT model from scratch
```
cd ~/kart/src
bash pretrain_bert_from_scratch.sh
```

#### 1-7. Pre-train BERT model from BERT-base-uncased
```
cd ~/kart/src
bash pretrain_bert_from_bert_base_uncased.sh
```

## Citation
Please cite our arXiv paper:

```
@misc{kart,
Author = {Yuta Nakamura and Shouhei Hanaoka and Yukihiro Nomura and Naoto Hayashi and Osamu Abe and Shuntaro Yada and Shoko Wakamiya and Eiji Aramaki},
Title = {KART: Privacy Leakage Framework of Language Models Pre-trained with Clinical Records},
Year = {2020},
Eprint = {arXiv:2101.00036},
}
```

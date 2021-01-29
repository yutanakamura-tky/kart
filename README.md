# KART: Privacy Leakage Framework of Language Models Pre-trained with Clinical Records
This is an implementation of our arXiv preprint paper (https://arxiv.org/abs/2101.00036) "KART: Privacy Leakage Framework of Language Models Pre-trained with Clinical Records."  

## Usage
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

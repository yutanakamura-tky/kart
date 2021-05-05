# How to check privacy risk of BERT models
### 2-1. Preparation

Split `MIMIC_III_DUMMY_PHI_HOSPITAL.csv` into six files:
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C1P2.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C1P1.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C1P0.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C0P2.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C0P1.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C0P0.csv`

```sh
cd ~/kart/src/modules/privacy
python split_mimic_iii_dummy_phi.py
```

Extract full name mentions from six CSV files above.  
A 'full name mention' refers to a span '(full name) is a (age) year-old (sex)' and subsequent five sentences.

```sh
python extract_full_name_mentions_from_mimic_iii_dummy_phi.py
```

Extract disease names from full name mentions.  
```sh
python extract_diseases_from_mimic_iii_dummy_phi.py
```

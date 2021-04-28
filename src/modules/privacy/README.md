# How to check privacy risk of BERT models
### 2-1. Convert CSV files:

```sh
cd ~/kart/src/modules/privacy
python extract_full_name_mentions.py
```

This will convert `MIMIC_III_DUMMY_PHI_HOSPITAL.csv` into six files:
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C1P2.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C1P1.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C1P0.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C0P2.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C0P1.csv`
- `MIMIC_III_DUMMY_PHI_HOSPITAL_C0P0.csv`
- 
- 
- 
- 

#!/bin/sh
#
# This script creates MIMIC-III-dummy-PIH,
# an authentic corpus of clinical records with dummy personal information from MIMIC-III clinical records.
#
# Usage
#
# 1. Prepare the following files:
#    - MIMIC-III noteevents at "../corpus/NOTEEVENTS.csv"
#    - i2b2 2006 de-identification training set at "../corpus/deid_surrogate_train.xml"
#
# 2. Run this script. This script will take about an hour to complete.
#
# 3. The following file will be generated:
#    - MIMIC-III-dummy-PHI noteevents at "../corpus/NOTEEVENTS_WITH_DUMMY_PHI.csv"


dataset_dir=../corpus
echo "===== Creation of MIMIC-III-dummy-PHI ====="

bash mimic_iii_dummy_phi/part_1_create_common_data.sh ${dataset_dir} 

bash mimic_iii_dummy_phi/part_2_create_surrogate_mapping.sh ${dataset_dir} hospital 42
bash mimic_iii_dummy_phi/part_2_create_surrogate_mapping.sh ${dataset_dir} shadow 12345

bash mimic_iii_dummy_phi/part_3_embed_dummy_phi_without_surrogate_mapping.sh ${dataset_dir}

bash mimic_iii_dummy_phi/part_4_embed_dummy_phi_with_surrogate_mapping.sh ${dataset_dir} hospital
bash mimic_iii_dummy_phi/part_4_embed_dummy_phi_with_surrogate_mapping.sh ${dataset_dir} shadow 

python mimic_iii_dummy_phi/part_5_concatenate_noteevents.py ${dataset_dir} hospital shadow


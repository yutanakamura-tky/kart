#!/bin/sh
#
# This script creates MIMIC-III-dummy-PIH,
# an authentic corpus of clinical records with dummy personal information from MIMIC-III clinical records.
# Reproducibility is confirmed.
#
# Usage
#
# 1. Prepare the following files:
#    - MIMIC-III noteevents at "../corpus/NOTEEVENTS.csv"
#    - i2b2 2006 de-identification training set at "../corpus/deid_surrogate_train_all_version2.xml"
#
# 2. Change working directory to "src".
#
# 3. Run this script. This script will take about an hour to complete.
#
# 4. The following file will be generated:
#    - MIMIC-III-dummy-PHI noteevents at "../corpus/NOTEEVENTS_WITH_DUMMY_PHI.csv"
#    - MIMIC-III-dummy-PHI hospital corpus at "../corpus/MIMIC_III_DUMMY_PHI_HOSPITAL.csv"
#    - MIMIC-III-dummy-PHI noteevents at "../corpus/MIMIC_III_DUMMY_PHI_SHADOW.csv"


dataset_dir=../corpus
log_path=make_mimic_iii_dummy_phi.log
echo "===== Creation of MIMIC-III-dummy-PHI ====="

bash modules/mimic_iii_dummy_phi/part_1_create_common_data.sh ${dataset_dir} 2>&1 | tee ${log_path} 

bash modules/mimic_iii_dummy_phi/part_2_create_surrogate_mapping.sh ${dataset_dir} hospital 42  2>&1 | tee ${log_path}
bash modules/mimic_iii_dummy_phi/part_2_create_surrogate_mapping.sh ${dataset_dir} shadow 12345 2>&1 | tee ${log_path}

bash modules/mimic_iii_dummy_phi/part_3_embed_dummy_phi_without_surrogate_mapping.sh ${dataset_dir} 2>&1 | tee ${log_path}

bash modules/mimic_iii_dummy_phi/part_4_embed_dummy_phi_with_surrogate_mapping.sh ${dataset_dir} hospital 2>&1 | tee ${log_path}
bash modules/mimic_iii_dummy_phi/part_4_embed_dummy_phi_with_surrogate_mapping.sh ${dataset_dir} shadow 2>&1 | tee ${log_path}

python modules/mimic_iii_dummy_phi/part_5_concatenate_noteevents.py ${dataset_dir} hospital shadow 2>&1 | tee ${log_path}
python modules/mimic_iii_dummy_phi/part_6_make_mimic_iii_dummy_phi_subsets.py ${dataset_dir}/NOTEEVENTS_WITH_DUMMY_PHI.csv 2>&1 | tee ${log_path}

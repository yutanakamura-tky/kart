#!/bin/bash
#
# Run this script via ../make_mimic_iii_dummy_phi.sh.
# Reproducibility Confirmed.

function replace_placeholders_with_dummy_phi_with_mapping() {

dataset_dir=$1
common_data_dir=${dataset_dir}/common_data
surrogate_map_dir=${dataset_dir}/dummy_phi/$2

tmp=${dataset_dir}/___temp_noteevents_text_with_dummy_phi.csv
output_pseudonymized_mimic_before_surrogate_mapping=${common_data_dir}/noteevents_text_with_dummy_phi_before_surrogate_mapping.csv
output_pseudonymized_mimic=${surrogate_map_dir}/noteevents_text_with_dummy_phi.csv
output_surrogate_map_with_id=${surrogate_map_dir}/surrogate_map_for_placeholders_with_id.csv

    cp ${output_pseudonymized_mimic_before_surrogate_mapping} ${tmp}
    echo -e "Copied to temporary file: ${output_pseudonymized_mimic_before_surrogate_mapping} -> ${tmp}"
    
    # New summary using placeholder map (but not placeholders without id)
    python mimic_iii_dummy_phi/placeholder_to_surrogate.py ${tmp} ${output_surrogate_map_with_id} ${tmp}_tmp
    mv ${tmp}_tmp ${output_pseudonymized_mimic}
    rm ${tmp}
    
    echo -e "Done! ${tmp} -> ${tmp}_tmp -> ${output_pseudonymized_mimic}"
}

save_root_dir=$1
save_rel_dir=$2

echo "===== 4. Embedding dummy personal information (2/2) (for the part that needs mapping) (${save_rel_dir} corpus)"
replace_placeholders_with_dummy_phi_with_mapping $save_root_dir $save_rel_dir

function create_common_output_files () {

dataset_dir=$1
common_data_dir=$2

tmp=${common_data_dir}/___temp_noteevents_text_cleaned.csv

input_mimic_file=${dataset_dir}/NOTEEVENTS.csv  # This data will be used repeatedly
input_i2b2_file=${dataset_dir}/deid_surrogate_train.xml  # This data will be used repeatedly
output_raw_noteevents=${common_data_dir}/noteevents_text_original.csv  # This data will be used repeatedly
output_cleaned_noteevents=${common_data_dir}/noteevents_text_cleaned.csv  # This data will be used repeatedly

output_unique_placeholder=${common_data_dir}/unique_placeholders.csv  # This data will be used repeatedly
output_placeholder_category=${common_data_dir}/placeholder_category.csv  # This data will be used repeatedly

output_surrogate_i2b2_age=${common_data_dir}/surrogate_age.csv
output_surrogate_i2b2_date=${common_data_dir}/surrogate_date.csv
output_surrogate_i2b2_doctor=${common_data_dir}/surrogate_doctor.csv
output_surrogate_i2b2_hospital=${common_data_dir}/surrogate_hospital.csv
output_surrogate_i2b2_id=${common_data_dir}/surrogate_id.csv
output_surrogate_i2b2_location=${common_data_dir}/surrogate_location.csv
output_surrogate_i2b2_patient=${common_data_dir}/surrogate_patient.csv
output_surrogate_i2b2_phone=${common_data_dir}/surrogate_phone.csv


# Extract noteevents text
echo -e "Extracting events in MIMIC-III ..."

python -c "
import pandas as pd

df = pd.read_csv('${input_mimic_file}', quoting=0, low_memory=False)
df['TEXT'].to_csv('${output_raw_noteevents}', header=False)
"
echo -e "Done! ${input_mimic_file} -> ${output_raw_noteevents}"



# Extract de-identification placeholders
echo -e "Extracting de-identification placeholders ..."
  
grep -oE "\[\*\*([^]])*?\*\*\]" ${output_raw_noteevents} | sort | uniq > ${output_unique_placeholder}
echo -e "Done! ${output_raw_noteevents} -> ${output_unique_placeholder}"


# Itemize all placeholder categories
echo -e "Gathering de-identification placeholders with the same categories ..."
cat ${output_unique_placeholder} | sed -E "s/^\[\*\*(.+) [0-9]*\*\*\]$/\1/g" | uniq | sort > ${output_placeholder_category}
echo -e "Done! ${output_unique_placeholder} -> ${output_placeholder_category}"


# Itemize all surrogate entities
echo -e "Collecting surrogate entities from i2b2 2006 corpus ..."
msg="Surrogate entities"

    grep -o -E '<PHI TYPE="AGE">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="AGE">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_age}
    echo -e "${msg} (1/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_age} (n=`uniq ${output_surrogate_i2b2_age} | wc -l`)"

    grep -o -E '<PHI TYPE="DATE">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="DATE">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_date}
    echo -e "${msg} (2/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_date} (n=`uniq ${output_surrogate_i2b2_date} | wc -l`)"

    grep -o -E '<PHI TYPE="DOCTOR">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="DOCTOR">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_doctor}
    echo -e "${msg} (3/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_doctor} (n=`uniq ${output_surrogate_i2b2_doctor} | wc -l`)"

    grep -o -E '<PHI TYPE="HOSPITAL">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="HOSPITAL">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_hospital}
    echo -e "${msg} (4/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_hospital} (n=`uniq ${output_surrogate_i2b2_hospital} | wc -l`)"

    grep -o -E '<PHI TYPE="ID">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="ID">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_id}
    echo -e "${msg} (5/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_id} (n=`uniq ${output_surrogate_i2b2_id} | wc -l`)"

    grep -o -E '<PHI TYPE="LOCATION">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="LOCATION">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_location}
    echo -e "${msg} (6/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_location} (n=`uniq ${output_surrogate_i2b2_location} | wc -l`)"

    grep -o -E '<PHI TYPE="PATIENT">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="PATIENT">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_patient}
    echo -e "${msg} (7/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_patient} (n=`uniq ${output_surrogate_i2b2_patient} | wc -l`)"

    grep -o -E '<PHI TYPE="PHONE">[^<]*?</PHI>' ${input_i2b2_file} | sed -E 's/<PHI TYPE="PHONE">(.*)<\/PHI>/\1/g' > ${output_surrogate_i2b2_phone}
    echo -e "${msg} (8/8): ${input_i2b2_file} -> ${output_surrogate_i2b2_phone} (n=`uniq ${output_surrogate_i2b2_phone} | wc -l`)"


###################################
# Processing MIMIC-III noteevents #
###################################


echo -e "Cleaning MIMIC-III noteevents ..."


    cp ${output_raw_noteevents} ${tmp}
    echo -e "Copied to temporary file: ${output_raw_noteevents} -> ${tmp}"
    
    # Replace "!" with "." 
    sed -i -E "s/[ ]*\!+/\./g" ${tmp}
    echo -e "Replaced '!' with '.' in ${tmp}"
    
    
    # Remove "??????" & Remove tabs
    sed -E "s/([0-9][ ]*)\?\?\?\?\?\?/\1/g" ${tmp} | tr -d '\t' | sed -E "s/^\?+//g" > "${tmp}_tmp"
    mv "${tmp}_tmp" ${tmp}
    echo -e "Removed '??????' and tabs in ${tmp}"
    
    
    # Delete "Dr. ......"
    sed -i -E "s/(Dr\.[ ]*)\.+//g" ${tmp}
    echo -e "Removed 'Dr. ......'"
    
    
    # Replace "[a-zA-Z]....." -> "[a-zA-Z]."
    sed -i -E "s/([a-zA-Z]\.)\.+/\1/g" ${tmp}
    echo -e "Replaced '[a-zA-Z]......' -> '[a-zA-Z]."
    
    sed -i -E "s/(\")\.\.+/\1/g" ${tmp}
    echo -e "Replaced '\".....' -> '\""
    
    sed -i -E "/^\.\.+$/d" ${tmp}
    echo -e "Removed lines with only '.'"
    
    sed -i -E "s/([a-zA-Z]\.)\.+/\1/g" ${tmp}
    
    
    # Remove "______"
    sed -i -E "/^[ ]*_+[ ]*$/d" ${tmp}
    echo -e "Removed lines with only '_'"
    
    sed -i -E "s/(grade|Edema \[.?\])[ ]_+/\1/g" ${tmp}
    sed -i -E "s/(Edema|Murmur)([^_]+)_*([^_]*)_+/\1\2\3/g" ${tmp} 
    sed -i -E "s/(last cigarette )_____/\1/g" ${tmp}
    echo -e "Removed '_'s at the end of checklists"
    
    sed -i -E "s/_+([0-9]+)_+[ ]*/\1 /g" ${tmp} 
    echo -e "Replaced '___[0-9]+___' -> '[0-9]+'"
    
    
    # Remove "~~~~~~"
    sed -i -E "/^~+$/d" ${tmp}
    sed -i -E "s/~~+//g" ${tmp}
    echo -e "Removed '~~'"
    
    
    # Remove lines that contains more than three consequtive"@"
    sed -i -E "/@@@/d" ${tmp}
    echo -e "Removed lines with more than three consequtive '@'"
    
    
    # Save cleaned file
    mv ${tmp} ${output_cleaned_noteevents}
    echo -e "Saved: ${tmp} -> ${output_cleaned_noteevents}"

}


function main () {

# Arguments
# $1: Directory to save output.
# $2: Random state. This will determine surrogate entities to be filled in deidentification placeholders.

dataset_dir=$1
common_data_dir=${dataset_dir}/common_data
random_state=$2

if [ -e ${dataset_dir} ]; then
    :
else
    mkdir ${dataset_dir}
    echo -e "Made output directory: ${dataset_dir}"
fi

if [ -e ${common_data_dir} ]; then
    :
else
    mkdir ${common_data_dir}
    echo -e "Made output directory: ${common_data_dir}"
fi

create_common_output_files ${dataset_dir} ${common_data_dir} 
}

rs=42
save_root_dir=$1

echo "===== 1. Creation of common data ====="
main $save_root_dir $rs

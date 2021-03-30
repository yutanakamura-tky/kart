#!/bin/bash
#
# Run this script via ../make_mimic_iii_dummy_phi.sh.
# Reproducibility Confirmed.

function create_surrogate_mapping () {

dataset_dir=$1
common_data_dir=${dataset_dir}/common_data
surrogate_map_dir=${dataset_dir}/dummy_phi/$2
random_state=$3


output_cleaned_noteevents=${common_data_dir}/noteevents_text_cleaned.csv  # This data will be used repeatedly

output_unique_placeholder=${common_data_dir}/unique_placeholders.csv  # This data will be used repeatedly

output_surrogate_i2b2_doctor=${common_data_dir}/surrogate_doctor.csv
output_surrogate_i2b2_hospital=${common_data_dir}/surrogate_hospital.csv

output_surrogate_key=${surrogate_map_dir}/surrogate_key.csv
output_surrogate_value=${surrogate_map_dir}/surrogate_value.csv
output_surrogate_map=${surrogate_map_dir}/surrogate_map.csv
output_surrogate_map_with_id=${surrogate_map_dir}/surrogate_map_for_placeholders_with_id.csv


    mkdir -p ${surrogate_map_dir}
    echo -e "Made surrogate map directory: ${surrogate_map_dir}"

echo -e "Creating surrogate mapping to ${output_surrogate_map} ..."

# Create mapping: [** **] -> Admission
grep "\[\*\* \*\*\]" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_admission.csv
cat ${surrogate_map_dir}/surrogate_key_admission.csv | wc -l | (\
    read x
    cmd="print('\n'.join(['Admission' for i in range($x)]))"
    python -c "$cmd") \
    > ${surrogate_map_dir}/surrogate_val_admission.csv
echo -e "Dummy identifiers: [** **] -> Admission"
paste -d"," ${surrogate_map_dir}/surrogate_key_admission.csv ${surrogate_map_dir}/surrogate_val_admission.csv > $output_surrogate_map


# Create mapping: Age over 90 -> integer [90, 115] 
grep "Age over 90" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_age.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_age.csv', '$output_surrogate_map', 'random_int', 90, 115, random_state=$random_state)
"



# Create mapping: Attending
grep -E "Attending" $output_unique_placeholder > ${surrogate_map_dir}/surrogate_key_attending.csv

cat $output_surrogate_i2b2_doctor | head -n `cat ${surrogate_map_dir}/surrogate_key_attending.csv | wc -l` > ${surrogate_map_dir}/surrogate_val_attending.csv

get_seeded_random()
{
  seed="$1";
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null;
}
seed=0;

# Option parsing adopted from https://stackoverflow.com/a/14203146
REST=""
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -s)
      seed="$2"
      shift
      shift
      ;;
    *)   # unknown option
      REST="$REST $1"
      shift # past argument
      ;;
  esac
done

shuf --random-source=<(get_seeded_random $random_state) ${surrogate_map_dir}/surrogate_val_attending.csv > ___temp.csv
mv ___temp.csv ${surrogate_map_dir}/surrogate_val_attending.csv

echo -e "Dummy identifiers: Attending -> i2b2 Doctor"
paste -d"," ${surrogate_map_dir}/surrogate_key_attending.csv ${surrogate_map_dir}/surrogate_val_attending.csv >> $output_surrogate_map




# Create mapping: CC Contact Info -> (Delete) 
grep -e "CC Contact Info" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_cc.csv
cat ${surrogate_map_dir}/surrogate_key_cc.csv | wc -l | (\
    read x
    cmd="print('\n'.join(['' for i in range($x)]))"
    python -c "$cmd") \
    > ${surrogate_map_dir}/surrogate_val_cc.csv
echo -e "Dummy identifiers: CC Contact Info -> (Nothing)"
paste -d"," ${surrogate_map_dir}/surrogate_key_cc.csv ${surrogate_map_dir}/surrogate_val_cc.csv >> $output_surrogate_map



# Create mapping: Address 
grep -e "Apartment Address(" -e "Street Address(" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_address.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_address.csv', '$output_surrogate_map', 'address', postprocess=lambda x: x.replace('\n', ' '), random_state=$random_state)
"

echo -e "Dummy identifiers: Apartment Address -> fake.address()" 
echo -e "Dummy identifiers: Street Address -> fake.address()" 


# Create mapping: Company
grep "Company" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_company.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_company.csv', '$output_surrogate_map', 'company', random_state=$random_state)
"

echo -e "Dummy identifiers: Company -> fake.company()"


# Create mapping: Country
grep "Country" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_country.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_country.csv', '$output_surrogate_map', 'country', random_state=$random_state)
"

echo -e "Dummy identifiers: Country -> fake.country()"



# Create mapping: E-mail
grep "E-mail" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_email.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_email.csv', '$output_surrogate_map', 'email', random_state=$random_state)
"


grep -E "^Email\:" ${output_cleaned_noteevents} \
    > ${surrogate_map_dir}/surrogate_key_email_2.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_email_2.csv', '$output_surrogate_map', 'email', postprocess=lambda x: f'Email: {x}', random_state=$random_state)
"

echo -e "Dummy identifiers: E-mail -> fake.email()"




# Create mapping: First name
grep -e "Doctor First Name" -e "\[\*\*First Name" -e "\[\*\* First Name" -e "Known firstname" -e "Name6 (MD)" -e "Name10 (NameIs)" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_first_name.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_first_name.csv', '$output_surrogate_map', 'first_name', random_state=$random_state)
"

echo -e "Dummy identifiers: Doctor First name -> fake.first_name()"
echo -e "Dummy identifiers: First name -> fake.first_name()"
echo -e "Dummy identifiers: Known firstname -> fake.first_name()"
echo -e "Dummy identifiers: Name6 (MD) -> fake.first_name()"
echo -e "Dummy identifiers: Name10 (NameIs) -> fake.first_name()"



# Create mapping: Last name
# Also "Dr.[**Name (NI) XXXXX**]" -> Dr. fake.last_name()
grep -e "Dictator Info" -e "Doctor Last Name" -e "\[\*\*Last Name" -e "Known lastname" -e "Name (STitle)" -e "Name5 (PTitle) [0-9]" -e "Name[78] (MD)" -e "Name11 (NameIs)" -e "Name13" -e "Name14" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_last_name.csv

grep -oE "Dr.[ ]?(\[\*\*Name \(NI\) [0-9]+\*\*\])" ${output_cleaned_noteevents} | sed -E "s/Dr.[ ]?(\[\*\*.*)/\1/g" | sort | uniq \
    >> ${surrogate_map_dir}/surrogate_key_last_name.csv

grep -oE "Dr.[ ]?(\[\*\*Name2 \(NI\) [0-9]+\*\*\])" ${output_cleaned_noteevents} | sed -E "s/Dr.[ ]?(\[\*\*.*)/\1/g" | sort | uniq \
    >> ${surrogate_map_dir}/surrogate_key_last_name.csv

grep -oE "Dr\.[ ]*\.+" ${output_cleaned_noteevents} | sort | uniq \
    >> ${surrogate_map_dir}/surrogate_key_last_name.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_last_name.csv', '$output_surrogate_map', 'last_name', random_state=$random_state)
"

echo -e "Dummy identifiers: Dictator Info -> fake.last_name()"
echo -e "Dummy identifiers: Doctor Last name -> fake.last_name()"
echo -e "Dummy identifiers: Last name -> fake.last_name()"
echo -e "Dummy identifiers: Known lastname -> fake.last_name()"
echo -e "Dummy identifiers: Name (STitle) -> fake.last_name()"
echo -e "Dummy identifiers: Name7 (MD) -> fake.last_name()"
echo -e "Dummy identifiers: Name8 (MD) -> fake.last_name()"
echo -e "Dummy identifiers: Name11 (NameIs) -> fake.last_name()"
echo -e "Dummy identifiers: Name13 (STitle) -> fake.last_name()"
echo -e "Dummy identifiers: Name14 (STitle) -> fake.last_name()"
echo -e "Dummy identifiers: 'Dr.' Name (IS) XXXXX -> 'Dr.' fake.last_name()"
echo -e "Dummy identifiers: 'Dr.' Name2 (IS) XXXXX -> 'Dr.' fake.last_name()"
echo -e "Dummy identifiers: Name5 (PTitle) XXXXX -> fake.last_name()"
echo -e "Dummy identifiers: 'Dr. .....' -> fake.last_name()"



# Create mapping: Female first name
grep "Female First Name" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_first_name_female.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_first_name_female.csv', '$output_surrogate_map', 'first_name_female', random_state=$random_state)
"

echo -e "Dummy identifiers: Female first name -> fake.first_name_female()"



# Create mapping: Male first name
grep "Male First Name" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_first_name_male.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_first_name_male.csv', '$output_surrogate_map', 'first_name_male', random_state=$random_state)
"

echo -e "Dummy identifiers: Male first name -> fake.first_name_male()"



# Create mapping: Name prefix
grep "Name Prefix (Prefixes)" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_prefix.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_prefix.csv', '$output_surrogate_map', 'prefix', random_state=$random_state)
"

echo -e "Dummy identifiers: Name prefix -> fake.prefix()"



# Create mapping: Name Initial
grep -e "Name12" -e "Initial (" -e "Initials (" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_initial.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_initial.csv', '$output_surrogate_map', 'first_name', postprocess=lambda x: x[0] + '.', random_state=$random_state)
"

echo -e "Dummy identifiers: Name Initial -> fake.first_name()[0] + '.'"
echo -e "Dummy identifiers: Name12 (NameIs) -> fake.first_name()[0] + '.'"
echo -e "Dummy identifiers: Initial -> fake.first_name()[0] + '.'"
echo -e "Dummy identifiers: Initials -> fake.first_name()[0] + '.'"



# Create mapping: Hospital
grep -E "Hospital[ 0-9][ *0-9]" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_hospital.csv

cat $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital \
    $output_surrogate_i2b2_hospital | head -n `cat ${surrogate_map_dir}/surrogate_key_hospital.csv | wc -l` \
    > ${surrogate_map_dir}/surrogate_val_hospital.csv

get_seeded_random()
{
  seed="$1";
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null;
}
seed=0;

REST=""
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -s)
      seed="$2"
      shift
      shift
      ;;
    *)   # unknown option
      REST="$REST $1"
      shift # past argument
      ;;
  esac
done

shuf --random-source=<(get_seeded_random $random_state) ${surrogate_map_dir}/surrogate_val_hospital.csv > ___temp.csv
mv ___temp.csv ${surrogate_map_dir}/surrogate_val_hospital.csv

echo -e "Dummy identifiers: Hospital -> i2b2 Hospital"
paste -d"," ${surrogate_map_dir}/surrogate_key_hospital.csv ${surrogate_map_dir}/surrogate_val_hospital.csv >> $output_surrogate_map



# Create mapping: Location (City Name)
grep -oE "\- \[\*\*Location([^]])+[0-9]+\*\*\]" ${output_cleaned_noteevents} | sed -E "s/\- (.+)/\1/g" | sort | uniq \
    > ${surrogate_map_dir}/surrogate_key_city.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_city.csv', '$output_surrogate_map', 'city', random_state=$random_state)
"

echo -e "Dummy identifiers: <Hospital Name> - Location -> <Hospital Name> - fake.city()"



# Create mapping: ID
grep -e "Clip Number" -e "Job Number" -e "MD Number" -e "Medical Record Number" -e "Serial Number" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_id.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_id.csv', '$output_surrogate_map', 'random_int', 1, 99999999, random_state=$random_state)
"

echo -e "Dummy identifiers: ID -> fake.random_int(1, 99999999)"


# Create mapping: Month -> %m/%d 
grep "January" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_jan.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_jan.csv', '$output_surrogate_map', 'random_int', 1, 31, postprocess=lambda x: f'1/{x}', random_state=$random_state)
"

echo -e "Dummy identifiers: Date (1)"


grep "February" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_feb.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_feb.csv', '$output_surrogate_map', 'random_int', 1, 28, postprocess=lambda x: f'2/{x}', random_state=$random_state)
"

echo -e "Dummy identifiers: Date (2)"



grep "March" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_mar.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_mar.csv', '$output_surrogate_map', 'random_int', 1, 31, postprocess=lambda x: f'3/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (3)"



grep "April" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_apr.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_apr.csv', '$output_surrogate_map', 'random_int', 1, 30, postprocess=lambda x: f'4/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (4)"



grep "May" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_may.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_may.csv', '$output_surrogate_map', 'random_int', 1, 31, postprocess=lambda x: f'5/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (5)"



grep "June" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_jun.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_jun.csv', '$output_surrogate_map', 'random_int', 1, 30, postprocess=lambda x: f'6/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (6)"



grep "July" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_jul.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_jul.csv', '$output_surrogate_map', 'random_int', 1, 31, postprocess=lambda x: f'7/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (7)"



grep "August" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_aug.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_aug.csv', '$output_surrogate_map', 'random_int', 1, 31, postprocess=lambda x: f'8/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (8)"



grep "September" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_sep.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_sep.csv', '$output_surrogate_map', 'random_int', 1, 30, postprocess=lambda x: f'9/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (9)"



grep "October" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_oct.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_oct.csv', '$output_surrogate_map', 'random_int', 1, 31, postprocess=lambda x: f'10/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (10)"



grep "November" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_nov.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_nov.csv', '$output_surrogate_map', 'random_int', 1, 30, postprocess=lambda x: f'11/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (11)"



grep "December" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md_dec.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md_dec.csv', '$output_surrogate_map', 'random_int', 1, 31, postprocess=lambda x: f'12/{x}', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (12)"




grep "Month (only)" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_month.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_month.csv', '$output_surrogate_map', 'month_name', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (13)"



grep -e "Month/Day (" -e "Month Day" -e "Day Month" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_md.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_md.csv', '$output_surrogate_map', 'date', '%m/%d', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (14)"



grep -e "Month Year" -e "Year/Month " -e "Month/Year" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_my.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_my.csv', '$output_surrogate_map', 'date', '%m/%y', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (15)"



grep -e "Month/Day/Year" -e "Year/Month/Day" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_ymd.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_ymd.csv', '$output_surrogate_map', 'date', '%m/%d/%y', random_state=$random_state)
"
echo -e "Dummy identifiers: Date (16)"



# Create mapping: Phone number
grep "Telephone/Fax" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_phone.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_phone.csv', '$output_surrogate_map', 'phone_number', random_state=$random_state)
"
echo -e "Dummy identifiers: Phone number -> fake.phone_number()"



# Create mapping: PO Box
grep "PO Box" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_po.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_po.csv', '$output_surrogate_map', 'random_int', postprocess=lambda x: f'PO Box {x}', random_state=$random_state)
"
echo -e "Dummy identifiers: PO Box -> fake.random_int()"



# Create mapping: Social Security Number
grep "Social Security Number" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_social_security.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_social_security.csv', '$output_surrogate_map', 'random_int', 0, 999999999, postprocess=lambda x: f'{x:09d}'[:3] + '-' + f'{x:09d}'[3:5] + '-' + f'{x:09d}'[5:], random_state=$random_state)
"
echo -e "Dummy identifiers: Social Security Number -> XXX-XX-XXXX"


# Create mapping: Medical record Number
grep "Medical record number" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_record.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_record.csv', '$output_surrogate_map', 'random_int', 0, 999999999, random_state=$random_state)
"
echo -e "Dummy identifiers: Medical Record Number -> XXX-XX-XXXX"


# Create mapping: State
grep "State " $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_state.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_state.csv', '$output_surrogate_map', 'state', random_state=$random_state)
"
echo -e "Dummy identifiers: State -> fake.state()"


# Create mapping: Unit No
grep -oE "Unit No:[ ]+\[\*\*Numeric Identifier [0-9]+\*\*\]" ${output_cleaned_noteevents} | sed -E "s/Unit No:[ ]+//g" | sort | uniq \
    > ${surrogate_map_dir}/surrogate_key_unit.csv

python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_unit.csv', '$output_surrogate_map', 'random_int', 0, 2000, random_state=$random_state)
"
echo -e "Dummy identifiers: Unit No: [**Numberic Identifier**] -> Unit No: (fake.random_int(0,2000))"



# Create mapping: University/College
grep "University" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_university.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_university.csv', '$output_surrogate_map', 'city', postprocess=lambda x: f'{x} University', random_state=$random_state)
"
echo -e "Dummy identifiers: University/College -> fake.city() + University"



# Create mapping: URL
grep "URL" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_url.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_url.csv', '$output_surrogate_map', 'url', postprocess=lambda x: x.replace('http://', '').replace('https://', '').replace('/', ''), random_state=$random_state)
"
echo -e "Dummy identifiers: URL -> fake.url().replace('http://', '').replace('https://', '').replace('/', '')"



# Create mapping: Zipcode
grep "State/Zipcode" $output_unique_placeholder \
    > ${surrogate_map_dir}/surrogate_key_zipcode.csv
python -c "
from mimic_iii_dummy_phi import add_surrogate_mapping
add_surrogate_mapping('${surrogate_map_dir}/surrogate_key_zipcode.csv', '$output_surrogate_map', 'zipcode', random_state=$random_state) 
"
echo -e "Dummy identifiers: Zipcode -> fake.zipcode()"


# Complement quotation
echo -e "Postprocessing ..."
sed -i "/^,$/d" ${output_surrogate_map}
sed -i -E '/\[\*\*.*\*\*\],([^"])+/s/(\[\*\*.*?\*\*\],)([^"]*)/\1\"\2\"/g' ${output_surrogate_map}
echo -e "Created surrogate mapping: ${output_surrogate_map}"

# Create surrogate mapping for only placeholders with ID
cp ${output_surrogate_map} ${output_surrogate_map_with_id}

sed -i '/^\[\*\* \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Age over 90 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Attending Info \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*CC Contact Info \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Apartment Address(1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Street Address(1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Street Address(2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Company \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Country \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*E-mail address \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\* First Name \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Doctor First Name \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name (STitle) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name (Titles) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name11 (Name Pattern1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name3 (LF) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name4 (NamePattern1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name5 (NamePattern1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name7 (NamePattern1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name8 (NamePattern2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name9 (NamePattern2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Known firstname \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name10 (NameIs) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name6 (MD) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Dictator Info \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Doctor Last Name (ambig) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Doctor Last Name \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*First Name (STitle) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Known lastname \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (LF) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (NamePattern1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (NamePattern4) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (NamePattern5) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (Prefixes) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (STitle) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (Titles) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (ambig) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name (un) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Last Name \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name (STitle) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name11 (NameIs) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name13 (STitle) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name14 (STitle) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name7 (MD) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name8 (MD) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Female First Name (ambig) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Female First Name (un) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Male First Name (un) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name Prefix (Prefixes) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Initial (NamePattern1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Initials (NamePattern4) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Initials (NamePattern5) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name Initial (MD) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name Initial (NameIs) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name Initial (PRE) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Name12 (NameIs) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Hospital \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Hospital1 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Hospital2 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Hospital3 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Hospital4 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Hospital5 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Hospital6 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Clip Number (Radiology) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Job Number \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*MD Number(1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*MD Number(2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*MD Number(3) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*MD Number(4) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Medical Record Number \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Serial Number\*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month (only) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Day Month \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Day Month Year \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month Day \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month Day Year (2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month\/Day (1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month\/Day (2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month\/Day (3) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month\/Day (4) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Day Month Year \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month\/Year (2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month\/Year 1 \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Year\/Month \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Month\/Day\/Year \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Year\/Month\/Day \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Telephone\/Fax (1) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Telephone\/Fax (2) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Telephone\/Fax (3) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Telephone\/Fax (5) \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*Social Security Number \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*State \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*University\/College \*\*\]/d' ${output_surrogate_map_with_id}
sed -i '/^\[\*\*URL \*\*\]/d' ${output_surrogate_map_with_id}
echo -e "Created surrogate mapping for placeholders with ID: ${output_surrogate_map_with_id}"


}

save_root_dir=$1
save_rel_dir=$2
rs=$3

echo -e "===== 2. Creation of mapping for dummy personal information (${save_rel_dir} corpus, random_state=${rs})====="
create_surrogate_mapping $save_root_dir $save_rel_dir $rs

function replace_placeholders_with_bummy_phi_without_mapping () {

dataset_dir=$1
common_data_dir=${dataset_dir}/common_data
random_state=$2

output_cleaned_noteevents=${common_data_dir}/noteevents_text_cleaned.csv  # This data will be used repeatedly

tmp=${dataset_dir}/___temp_noteevents_text_with_dummy_phi_before_surrogate_mapping.csv
output_pseudonymized_mimic_before_surrogate_mapping=${common_data_dir}/noteevents_text_with_dummy_phi_before_surrogate_mapping.csv

echo -e "Replacing placeholders that can be handled without surrogate mapping ..."


    cp ${output_cleaned_noteevents} ${tmp}
    echo -e "Copied to temporary file: ${output_cleaned_noteevents} -> ${tmp}"
    
    # Replace ***day [**m-d**] -> **day m/d
    sed -i -E 's/(SUNDAY,?[ ]*|[Ss]unday,?[ ]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g' ${tmp}
    echo -e "Replaced placeholders: Sunday [**m-d**] -> Sunday m/d"
    
    sed -i -E 's/(MONDAY,?[ ]*|[Mm]onday,?[ ]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g' ${tmp}
    echo -e "Replaced placeholders: Monday [**m-d**] -> Monday m/d"
    
    sed -i -E 's/(TUESDAY,?[ ]*|[Tt]uesday,?[ ]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g' ${tmp}
    echo -e "Replaced placeholders: Tuesday [**m-d**] -> Tuesday m/d"
    
    sed -i -E 's/(WEDNESDAY,?[ ]*|[Ww]ednesday,?[ ]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g' ${tmp}
    echo -e "Replaced placeholders: Wednesday [**m-d**] -> Wednesday m/d"
    
    sed -i -E 's/(THURDSAY,?[ ]*|[Tt]hursday,?[ ]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g' ${tmp}
    echo -e "Replaced placeholders: Thursday [**m-d**] -> Thursday m/d"
    
    sed -i -E 's/(FRIDAY,?[ ]*|[Ff]riday,?[ ]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g' ${tmp}
    echo -e "Replaced placeholders: Friday [**m-d**] -> Friday m/d"
    
    sed -i -E 's/(SATURDAY,?[ ]*|[Ss]aturday,?[ ]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g' ${tmp}
    echo -e "Replaced placeholders: Saturday [**m-d**] -> Saturday m/d"
    
    
    
    # Replace <medical exam> [**m-d**] -> <medical exam> m/d
    sed -i -E "s/(bal[^\.A-Z]+?|graphy?[^\.A-Z]+?|xr[^\.A-Z]+?|kub[^\.A-Z]+?|echo[^\.A-Z]+?|e[cek]g[^\.A-Z]+?|t[te]e[^\.A-Z]+?)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    
    sed -i -E "s/(BAL[^\.A-Z]+?|GRAPHY?[^\.A-Z]+?|[Cc]ulture[^\.A-Z]+?|CSF[^\.A-Z]+?|EGD[^\.A-Z]+?|E[CEK]G[^\.A-Z]+?)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    
    sed -i -E "s/(CTAP[^\.A-Z]+?|CT[AU]?[^\.A-Z]+?|MR[ACI][P]?[^\.A-Z]+?|XR[^\.A-Z]+?|[Pp][Oo][Rr][Tt][Aa][Bb][Ll][Ee][A-Za-z ]*|[S]?PE[C]?T[^\.A-Z]+?|KUB[^\.A-Z]+?|ECHO[^\.A-Z]+?|US[^\.A-Z]+?|T[TE]E[^\.A-Z]+?)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    
    sed -i -E "s/(Echo[^\.A-Z]+?)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    
    sed -i -E "s/([A-Za-z]+opy[ ]+|[A-Za-z]+omy[ ]+|[A-Za-z]+opsy[ ]+)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    
    echo -e "Replaced placeholders: <medical exam> [**m-d**] -> <medical exam> m/d"
    
    
    sed -i -E "s/\[\*\*([0-9]+)-([0-9]+)\*\*\]([ ]+xr|[ ]+kub|[ ]+echo|[ ]+e[ck]g|[ ]+t[te]e)/\1\/\2\3/g" ${tmp} 
    
    sed -i -E "s/\[\*\*([0-9]+)-([0-9]+)\*\*\]([ ]+BAL|[ ]+CSF?|[ ]+EGD|[ ]+E[ECK]G)/\1\/\2\3/g" ${tmp} 
    
    sed -i -E "s/\[\*\*([0-9]+)-([0-9]+)\*\*\]([:,]? [a ]?[the ]?[A-Za-z]+[ ]*)([Cc]ulture|graph)/\1\/\2\3\4/g" ${tmp}
    
    sed -i -E "s/\[\*\*([0-9]+)-([0-9]+)\*\*\]([ ]+CT[AU]?|[ ]+MR[ACI][P]?|[ ]+[C]?XR|[^.,]*[Pp][Oo][Rr][Tt][Aa][Bb][Ll][Ee]|[ ]+[S]?PE[C]?T|[ ]+KUB|[ ]+ECHO|[ ]+US|[ ]+T[TE]E)/\1\/\2\3/g" ${tmp}
    
    sed -i -E "s/\[\*\*([0-9]+)-([0-9]+)\*\*\]([ ]+[A-Za-z]+opy|[ ]+[A-Za-z]+omy|[ ]+[A-Za-z]+opsy)/\1\/\2\3/g" ${tmp}
    
    echo -e "Replaced placeholders: [**m-d**] <medical exam> -> m/d <medical exam>"
    
    
    
    
    # Replace <prep> [**m-d**] -> <prep> m/d
    # Replace End date [**m-d**] -> End date m/d
    sed -i -E "s/([Dd]ated[ ]+|[Oo]n[ ]+|[Oo]n[ ]+the |[Ff]rom[ ]+|[Ss]ince[ ]+|[Tt]hrough[ ]+|[Tt]hru[ ]+|[Tt]rough[ ]+|[Tt]o[ ]+|[Uu]ntil[ ]+)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    sed -i -E "s/(DATED[ ]+|ON[ ]+|FROM[ ]+|SINCE[ ]+|THROUGH[ ]+|THRU[ ]+|TROUGH[ ]+|TO[ ]+|UNTIL[ ]+)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    sed -i -E "s/([Mm]orning[ ]+|[Aa]fternoon[ ]+|[Ee]vening[ ]+|[Nn]ight[ ]+)([Oo][Ff][ ]+)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\3\/\4/g" ${tmp}
    sed -i -E "s/(MORNING[ ]+|AFTERNOON[ ]+|EVENING[ ]+|NIGHT[ ]+)([Oo][Ff][ ]+)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\3\/\4/g" ${tmp}
    
    echo -e "Replaced placeholders: <prep> [**m-d**] -> <prep> m/d"
    
    sed -i -E "s/\[\*\*([0-9]+)-([0-9]+)\*\*\]([ ]+[Aa][Tt][ ]+[0-9]+)/\1\/\2\3/g" ${tmp}
    echo -e "Replaced placeholders: [**m-d**] <prep> -> m/d <prep>"
    
    sed -i -E "s/([Ee][Nn][Dd] [Dd][Aa][Tt][Ee][^.,]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp} 
    echo -e "Replaced placeholders: End date [**m-d**] -> End date"
    
    sed -i -E "s/([Ss][Tt][Aa][Rr][Tt] [Dd][Aa][Tt][Ee][^.,]*)\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g" ${tmp}
    echo -e "Replaced placeholders: Start date [**m-d**] -> Start date"
    
    sed -i -E "s/([0-9]+\/[0-9]+ and )\[\*\*([0-9]+)-([0-9]+)\*\*\]/\1\2\/\3/g"  ${tmp}
    echo -e "Replaced placeholders: <prep> m1/d1 and [**m2-d2**] -> <prep> m1/d1 and m2/d2"
    
    
    # Replace [**m-d**] at line heads -> m/d
    sed -i -E "s/^\[\*\*([0-9]+)-([0-9]+)\*\*\](.*)$/\1\/\2\3/g" ${tmp}
    echo -e "Replaced placeholders: [**m-d**] at line heads -> m/d"
    
    
    
    
    
    # Replace in [**XX-YY**] day -> in ZZ day
    # Rule 1. For ranges including 14 <= day -> in 14 day
    # Rule 2. For ranges including 10 <= day < 14  -> in 10 day
    # Rule 3. For ranges including 7 <= day < 10 -> in 7 day
    # Rule 4. For ranges including 5 <= day < 7 -> in 5 day
    # Rule 5. For ranges including 3 <= day < 4 -> in 3 day
    # Rule 6. Other ranges -> in (the largest integer in the range) week
    sed -i -E "s/in \[\*\*([0-9]|1[0-2])-(1[4-9]|[2-3][0-9])\*\*\] day/in 14 day/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] day -> in 14 day for x>=14 or y>=14"
    
    sed -i -E "s/in \[\*\*([0-9]-1[0-3]|1[0-2]-[0-9]|1[0-2]-1[0-3])\*\*\] day/in 10 day/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] day -> in 10 day for 10<=x<14 and 10<=y<14"
    
    sed -i -E "s/in \[\*\*([0-6]-[7-9]|[7-9]-[0-9])\*\*\] day/in 7 day/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] day -> in 7 day for 7<=x<10 and 7<=y<10"
    
    sed -i -E "s/in \[\*\*([0-2]-[3-4]|[3-4]-[0-2])\*\*\] day/in 3 day/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] day -> in 3 day for 3<=x<7 and 3<=y<7"
    
    sed -i -E "s/in \[\*\*(1-2|2-2|2-1)\*\*\] day/in 2 day/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] day -> in 2 day"
    
    sed -i -E "s/in \[\*\*(1-1)\*\*\] day/in 1 day/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] day -> in 1 day"
    
    
    
    # Replace in [**XX-YY**] week -> in ZZ week
    # Rule 1. For ranges including 24 <= week -> in 24 week
    # Rule 2. For ranges including 12 <= week < 24  -> in 12 week
    # Rule 3. For ranges including 8 <= week < 11 -> in 8 week
    # Rule 4. Other ranges -> in (the largest integer in the range) week
    sed -i -E "s/in \[\*\*([1-9]|1[0-2])-(2[4-9]|3[0-9])\*\*\] week/in 24 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] week -> in 24 week for x>=24 or y>=24"
    
    sed -i -E "s/in \[\*\*12-([0-9]|1[0-9]|2[0-3])\*\*\] week/in 12 week/g" ${tmp}
    sed -i -E "s/in \[\*\*([1-9]|1[0-1])-(1[2-9]|2[0-3])\*\*\] week/in 12 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**x-y**] week -> in 12 week for 12<=x<24 and 12<=y<24"
    
    sed -i -E "s/in \[\*\*([1-9]-[8-9]|[1-9]-1[0-1]|[8-9]-[0-7])\*\*\] week/in 8 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] week -> in 8 week for 8<=x<12 and 8<=y<12"
    
    sed -i -E "s/in \[\*\*1-([1-7])\*\*\] week/in \1 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**1-y**] week -> in y week for y<8 and x<=y"
    
    sed -i -E "s/in \[\*\*2-([2-7])\*\*\] week/in \1 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**2-y**] week -> in y week for y<8 and x<=y"
    
    sed -i -E "s/in \[\*\*3-([3-7])\*\*\] week/in \1 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**3-y**] week -> in y week for y<8 and x<=y"
    
    sed -i -E "s/in \[\*\*4-([4-7])\*\*\] week/in \1 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**4-y**] week -> in y week for y<8 and x<=y"
    
    sed -i -E "s/in \[\*\*5-([5-7])\*\*\] week/in \1 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**5-y**] week -> in y week for y<8 and x<=y"
    
    sed -i -E "s/in \[\*\*6-([6-7])\*\*\] week/in \1 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**6-y**] week -> in y week for y<8 and x<=y"
    
    sed -i -E "s/in \[\*\*7-([7-7])\*\*\] week/in \1 week/g" ${tmp} 
    echo -e "Replaced placeholders: in [**7-y**] week -> in y week for y<8 and x<=y"
    
    sed -i -E "s/in \[\*\*(2)-([1-1])\*\*\] week/in \1 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**2-y**] week -> in 2 week for x>y"
    
    sed -i -E "s/in \[\*\*(3)-([1-2])\*\*\] week/in \1 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**3-y**] week -> in 3 week for x>y"
    
    sed -i -E "s/in \[\*\*(4)-([1-3])\*\*\] week/in \1 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**4-y**] week -> in 4 week for x>y"
    
    sed -i -E "s/in \[\*\*(5)-([1-4])\*\*\] week/in \1 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**5-y**] week -> in 5 week for x>y"
    
    sed -i -E "s/in \[\*\*(6)-([1-5])\*\*\] week/in \1 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**6-y**] week -> in 6 week for x>y"
    
    sed -i -E "s/in \[\*\*(7)-([1-6])\*\*\] week/in \1 week/g" ${tmp}
    echo -e "Replaced placeholders: in [**7-y**] week -> in 7 week for x>y"
    
    
    
    # Replace in [**X-Y**] month -> in Z month
    # Rule 1. For ranges including 12 <= month -> in 12 month
    # Rule 2. For ranges including 6 <= month < 12  -> in 6 month
    # Rule 3. For ranges including 3 <= month < 6 -> in 3 month
    # Rule 4. Other ranges -> in (the largest integer in the range) month
    
    sed -i -E "s/in \[\*\*([0-9]|1[0-1])-(1[2-9]|[2-3][0-9])\*\*\] month/in 12 month/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] month -> in 12 month"
    
    sed -i -E "s/in \[\*\*([0-5])-([6-9]|1[0-1])\*\*\] month/in 6 month/g" ${tmp}
    sed -i -E "s/in \[\*\*([6-9]|1[0-1])-([0-9]+)\*\*\] month/in 6 month/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] month -> in 6 month"
    
    sed -i -E "s/in \[\*\*([1-3]-[3-5]|[3-5]-[1-3])\*\*\] month/in 3 month/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] month -> in 3 month"
    
    sed -i -E "s/in \[\*\*(1-2|2-1|2-2)\*\*\] month/in 2 month/g" ${tmp}
    echo -e "Replaced placeholders: in [**x-y**] month -> in 2 month"
    
    sed -i -E "s/in \[\*\*(1-1)\*\*\] month/in 1 month/g" ${tmp} 
    echo -e "Replaced placeholders: in [**x-y**] month -> in 1 month"
    
    
    # Replace "Completed by:[**0-0-0**]" -> "Completed by:"
    sed -i -E "s/(Completed by:)\[\*\*0-0-0\*\*\]/\1/g" ${tmp}
    echo -e "Replaced placeholders: 'Completed by:[**0-0-0**]' -> 'Completed by:'"
    
    
    # Replace CA [**XX**]-9 -> CA 19-9
    sed -i -E "s/\[\*\*[0-9]+\*\*\]-9 /19-9 /g" ${tmp}
    echo -e "Replaced placeholders: CA [**XX**]-9 -> CA 19-9"
    
    mv ${tmp} ${output_pseudonymized_mimic_before_surrogate_mapping}
    echo -e "Done! ${tmp} -> ${output_pseudonymized_mimic_before_surrogate_mapping}"

    rm ${dataset_dir}/sed*
    
}

dir_corpus=$1

echo "===== 3. Embedding dummy personal information (1/2) (for the part where mapping is not needed) ====="
replace_placeholders_with_bummy_phi_without_mapping $dir_corpus

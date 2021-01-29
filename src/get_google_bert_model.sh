#!/bin/sh
#
# This script will download the uncase BERT-base model by Google Research.
# The model will be saved at "../model/bert".

dir_model=../model/bert
path_zip=$dir_model/uncased_L-12_H-768_A-12.zip

echo "===== Getting non-domain-specific uncased BERT-base model ====="

if [ -e ${dir_model} ]; then
    :
else
    mkdir -p ${dir_model}
    echo "Made a new directory: ${dir_model}"
fi

wget -nH --cut-dirs=3 https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -O $path_zip
unzip $path_zip -d $dir_model

rm $path_zip
echo -e "Downloading & unzipping uncased BERT-base model successful!"
echo -e "Removed ${path_zip}"

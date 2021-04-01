#!/bin/sh
#
# NOTE: This script will take around four hours to complete.
#
# USAGE
#
# This script converts pretraining corpus txt file into tfrecord file.
#
# Run this script as below:
# 1. To convert D_private or D_public (Hospital corpus)
#     1-1. Large subset (c1p2)
#         bash corpus_to_tfrecord.sh hospital c1p2 hipaa
#         bash corpus_to_tfrecord.sh hospital c1p2 no_anonymization
#     1-2. Small subset (c0p2)
#         bash corpus_to_tfrecord.sh hospital c0p2 hipaa
#         bash corpus_to_tfrecord.sh hospital c0p2 no_anonymization
#     1-3. c1p1
#         bash corpus_to_tfrecord.sh hospital c1p1 hipaa
#         bash corpus_to_tfrecord.sh hospital c1p1 no_anonymization
#     1-4. c0p1
#         bash corpus_to_tfrecord.sh hospital c0p1 hipaa
#         bash corpus_to_tfrecord.sh hospital c0p1 no_anonymization
#     1-5. c1p0
#         bash corpus_to_tfrecord.sh hospital c1p0 hipaa
#         bash corpus_to_tfrecord.sh hospital c1p0 no_anonymization
#     1-6. c0p0
#         bash corpus_to_tfrecord.sh hospital c0p0 hipaa
#         bash corpus_to_tfrecord.sh hospital c0p0 no_anonymization
#
# 2. To convert D_shadow (Shadow corpus)
#         bash corpus_to_tfrecord.sh shadow

corpus=$1      # Either of hospital, shadow
code=$2        # Either of c1p2, c0p2, c1p1, c0p1, c1p0, c0p0
anonymity=$3   # Either of hipaa, no_anonymization

root_dir=../../..
dir_bert_model=${root_dir}/model/bert
dir_bert_tf1=../bert
dir_corpus=../corpus/pretraining_corpus

script_path=${dir_bert_tf1}/create_pretraining_data.py

basename_corpus=pretraining_corpus_${1}_${2}_${3}.txt
input_file=${dir_corpus}/${basename_corpus}
vocab_file=${dir_bert_model}/vocab.txt

basename_tfrecord_128=tf_examples_${1}_${2}_${3}_128.tfrecord
basename_tfrecord_512=tf_examples_${1}_${2}_${3}_512.tfrecord
output_file_128=${dir_corpus}/${basename_tfrecord_128}
output_file_512=${dir_corpus}/${basename_tfrecord_512}



### Corpora -> tfrecord
# Generate datasets for 128 max seq
python ${script_path} \
  --input_file=${input_file} \
  --output_file=${output_file_128} \
  --vocab_file=${vocab_file} \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3

# # Generate datasets for 512 max seq
# python ${script_path} \
#   --input_file=${input_file} \
#   --output_file=${output_file_512} \
#   --vocab_file=${vocab_file} \
#   --do_lower_case=True \
#   --max_seq_length=512 \
#   --max_predictions_per_seq=76 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=3
# 

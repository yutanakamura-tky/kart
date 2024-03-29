#!/bin/sh
#
# NOTE: This script will take around four hours to complete.
#
# USAGE
#
# This script converts pretraining corpus txt file into tfrecord file.
#
# Run this script as below:
# 1. To convert D_private or D_public (Hospital corpus) to tfrecord file:
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
# 2. To convert D_shadow (Shadow corpus) to tfrecord file:
#         bash corpus_to_tfrecord.sh shadow

corpus=$1      # Either of hospital, shadow
code=$2        # Either of c1p2, c0p2, c1p1, c0p1, c1p0, c0p0
anonymity=$3   # Either of hipaa, no_anonymization

root_dir=..
dir_bert_model=${root_dir}/models/bert
dir_bert_tf1=${root_dir}/src/modules/bert
dir_corpus=${root_dir}/corpus/pretraining_corpus/${1}

script_path=${dir_bert_tf1}/create_pretraining_data.py

basename_corpus=pretraining_corpus_${2}_${3}.txt
input_file=${dir_corpus}/${basename_corpus}
vocab_file=${dir_bert_model}/vocab.txt

stem_tfrecord_128=tf_examples_${2}_${3}_128
stem_tfrecord_512=tf_examples_${2}_${3}_512
basename_tfrecord_128=${stem_tfrecord_128}.tfrecord
basename_tfrecord_512=${stem_tfrecord_512}.tfrecord
basename_tfrecord_128_wwm=${stem_tfrecord_128}_wwm.tfrecord
basename_tfrecord_512_wwm=${stem_tfrecord_512}_wwm.tfrecord

output_file_128=${dir_corpus}/${basename_tfrecord_128}
output_file_512=${dir_corpus}/${basename_tfrecord_512}
output_file_128_wwm=${dir_corpus}/${basename_tfrecord_128_wwm}
output_file_512_wwm=${dir_corpus}/${basename_tfrecord_512_wwm}



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

# Generate datasets for 128 max seq with WWM
python ${script_path} \
  --input_file=${input_file} \
  --output_file=${output_file_128_wwm} \
  --vocab_file=${vocab_file} \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3 \
  --do_whole_word_mask=True

# Generate datasets for 512 max seq
python ${script_path} \
  --input_file=${input_file} \
  --output_file=${output_file_512} \
  --vocab_file=${vocab_file} \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3
 
# Generate datasets for 512 max seq WWM
python ${script_path} \
  --input_file=${input_file} \
  --output_file=${output_file_512_wwm} \
  --vocab_file=${vocab_file} \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3 \
  --do_whole_word_mask=True

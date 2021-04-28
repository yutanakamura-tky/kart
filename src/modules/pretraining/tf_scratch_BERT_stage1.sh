#!bin/sh
#
# This is a modification from the notebook by Kexin Huang: 
# https://github.com/kexinhuang12345/clinicalBERT/blob/master/notebook/pretrain.ipynb

label_corpus=$1    # hospital or shadow
label_diversity=$2 # c1p2, c1p1, c1p0, c0p2, c0p1, c0p0
label_anonymization=$3 # hipaa or no_anonymization

dir_root=..
dir_corpus=${dir_root}/corpus/pretraining_corpus/${label_corpus}

dir_models=${dir_root}/models
dir_pretrained_bert=${dir_models}/bert

dir_script=${dir_root}/src/modules/bert

dir_model_save=${dir_models}/tf_bert_scratch_${label_corpus}_${label_diversity}_${label_anonymization}

if [ -e ${dir_model_save} ]; then
    :
else
    mkdir ${dir_model_save}
fi

CUDA_VISIBLE_DEVICES=0 python ${dir_script}/run_pretraining.py \
  --input_file=${dir_corpus}/tf_examples_${label_diversity}_${label_anonymization}_128.tfrecord \
  --output_dir=${dir_model_save}/pretraining_output_stage1 \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=${dir_pretrained_bert}/bert_config.json \
  --train_batch_size=64 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5



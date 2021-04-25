# Execute this script in model_performance_check dir.

model_dir=../../models
ner_dir=../../biomedical_ner

corpus_dir=${ner_dir}/corpus/ner/2006_deidentification
use_corpus="--train-dirs ${corpus_dir}/train --val-dirs ${corpus_dir}/develop --test-dirs ${corpus_dir}/test"

### Hyperparameters #1

hparam_4="--batch-size 16 -e 20 --fine-tune-biobert --lr-biobert 5e-5"
prefix="i2b2_2006_deid"
suffix_4="lr_5e-5_1e-2_epoch_20_batch_4"

python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
   ${use_corpus} \
    --model biobert \
    --biobert-path ${model_dir}/tf_bert_hospital_c1p2_1M/pretraining_output_stage1/model.ckpt-100000 \
    ${hparam_4} \
    -v c1p2_1M_${prefix}_${suffix_4}

python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
   ${use_corpus} \
    --model biobert \
    --biobert-path ${model_dir}/bert/bert_model.ckpt \
    ${hparam_4} \
    -v general_${prefix}_${suffix_4}

python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
    ${use_corpus} \
    --model biobert \
    --biobert-path ${model_dir}/tf_bert_hospital_c0p2_100k/pretraining_output_stage1/model.ckpt-100000 \
    ${hparam_4} \
    -v c0p2_100k_${prefix}_${suffix_4}

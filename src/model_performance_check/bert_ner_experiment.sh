model_dir=../../models
ner_dir=../../biomedical_ner
ncbi_dir=${ner_dir}/corpus/ncbi_disease_corpus
use_ncbi="--train-dirs ${ncbi_dir}/train --val-dirs ${ncbi_dir}/develop --test-dirs ${ncbi_dir}/test"

### Hyperparameters #1

# c1p2_1M_ncbi_lr_1e-2_epoch_15_batch_16.log

# python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
#    ${use_ncbi} \
#     --model biobert \
#     --biobert-path ${model_dir}/tf_bert_hospital_c1p2_1M/pretraining_output_stage1/model.ckpt-100000 \
#     -c 0


### Hyperparameters #2

# c1p2_1M_ncbi_lr_1e-2_epoch_15_batch_32.log
# python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
#    ${use_ncbi} \
#     --model biobert \
#     --biobert-path ${model_dir}/tf_bert_hospital_c1p2_1M/pretraining_output_stage1/model.ckpt-100000 \
#     -c 0 \
#     --batch-size 32


### Hyperparameters #3

# c1p2_1M_ncbi_lr_1e-2_epoch_30_batch_32.log
# python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
#    ${use_ncbi} \
#     --model biobert \
#     --biobert-path ${model_dir}/tf_bert_hospital_c1p2_1M/pretraining_output_stage1/model.ckpt-100000 \
#     -c 0 \
#     --batch-size 32 \
#     -e 30

# general_ncbi_lr_1e-2_epoch_30_batch_32.log
# python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
#    ${use_ncbi} \
#     --model biobert \
#     --biobert-path ${model_dir}/bert/bert_model.ckpt \
#     -c 0 \
#     --batch-size 32 \
#     -e 30

# c0p2_100k_ncbi_lr_1e-2_epoch_30_batch_32.log
python ${ner_dir}/src/modules/ner/train_ner_tagger.py \
    ${use_ncbi} \
    --model biobert \
    --biobert-path ${model_dir}/tf_bert_hospital_c0p2_100k/pretraining_output_stage1/model.ckpt-100000 \
    -c 0 \
    --batch-size 32 \
    -e 30


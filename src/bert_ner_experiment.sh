ner_dir=../biomedical_ner

python ${ner_dir}/src/modules/ner/train_ner_tagger.py --train-dirs ${ner_dir}/corpus/ncbi_disease_corpus/train --val-dirs ${ner_dir}/corpus/ncbi_disease_corpus/develop --test-dirs ${ner_dir}/corpus/ncbi_disease_corpus/test --model biobert --biobert-path ../models/tf_bert_hospital_c1p2_1M/pretraining_output_stage1/model.ckpt-100000 -c 0

python modules/pretraining/noteevents_to_pretraining_corpus.py ../corpus

# bash modules/pretraining/corpus_to_tfrecord.sh hospital c1p2 no_anonymization
# bash modules/pretraining/corpus_to_tfrecord.sh hospital c1p2 hipaa
bash modules/pretraining/corpus_to_tfrecord.sh hospital c0p2 no_anonymization
bash modules/pretraining/corpus_to_tfrecord.sh hospital c0p2 hipaa

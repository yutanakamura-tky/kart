import argparse
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

import nltk
import pandas as pd
from tqdm import tqdm

from kart.src.modules.privacy.utils.path import get_repo_dir
from kart.src.modules.pymetamap import pymetamap


def main():
    args = get_args()
    input_path = (
        get_repo_dir()
        / f"corpus/MIMIC_III_DUMMY_PHI_{args.corpus.upper()}_{args.code.upper()}.csv"
    )
    output_path = (
        get_repo_dir() / f"corpus/gold_disease_names_{args.corpus}_{args.code}.tsv"
    )

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path, quoting=0)

    full_name_mentions = df.query("patient_full_name_tfreq>0")[
        "full_name_mention"
    ].values
    patient_ages = df.query("patient_full_name_tfreq>0")["patient_age"].values
    patient_full_names = df.query("patient_full_name_tfreq>0")[
        "patient_full_name"
    ].values

    mm = pymetamap.MetaMap.get_instance(
        "/home/nakamura/metamap/public_mm/bin/metamap20",
    )

    texts = [text.replace("\n", "") for text in full_name_mentions]

    option = {"restrict_to_sts": ["dsyn", "mobd", "neop"]}

    results_with_option = extract_umls_concepts(texts, mm, option)

    print("Extracting disease names ...")
    diseases = []

    for i in tqdm(range(len(full_name_mentions))):
        for entity in results_with_option[i]:
            superficials = entity["superficials"]
            concept = entity["concept"]
            disease = OrderedDict(
                [
                    ("document_id", i),
                    ("patient_full_name", " ".join(eval(patient_full_names[i]))),
                    ("patiant_age", patient_ages[i]),
                    ("superficials", ", ".join(superficials)),
                    ("score", concept.score),
                    ("semtypes", concept.semtypes),
                    ("cui", concept.cui),
                    ("perferred_name", concept.preferred_name),
                ]
            )
            diseases.append(disease)

    df_diseases = pd.DataFrame(diseases)
    df_diseases.to_csv(output_path, index=False, sep="\t")
    print(f"Disease name saved to {output_path}")


def extract_umls_concepts(
    texts: List[str], mm: pymetamap.SubprocessBackend, option: Dict = {}
) -> List[List[Dict]]:
    result = [
        extract_umls_concepts_from_one_text(text, mm, option) for text in tqdm(texts)
    ]
    return result


def extract_umls_concepts_from_one_text(
    text: str, mm: pymetamap.SubprocessBackend, option: Dict = {}
) -> List[Dict]:
    sentences = text_to_sentences(text)
    concepts, error = mm.extract_concepts(sentences, range(len(sentences)), **option)
    entities = convert_concepts_to_superficial_concept_pair(concepts, sentences)
    return entities


def text_to_sentences(text: str) -> List[str]:
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentences = tokenizer.sentences_from_text(text)
    return sentences


def convert_concepts_to_superficial_concept_pair(
    concepts: List[pymetamap.Concept.ConceptMMI], sentences: List[str]
) -> List[Dict]:
    entities = []
    for concept in concepts:
        sentence_id = int(concept.index)
        raw_pos_info: List[Tuple[str, str]] = re.findall(
            r"(\d+)/(\d+)", concept.pos_info
        )
        positions = [
            {
                "start": int(pos_info[0]) - 1,
                "end": int(pos_info[0]) + int(pos_info[1]) - 1,
            }
            for pos_info in raw_pos_info
        ]
        superficials = [
            sentences[sentence_id][pos["start"] : pos["end"]] for pos in positions
        ]
        entities.append({"superficials": superficials, "concept": concept})
    return entities


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", choices=["hospital", "shadow"])
    parser.add_argument(
        "code", choices=["c1p2", "c1p1", "c1p0", "c0p2", "c0p1", "c0p0"]
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

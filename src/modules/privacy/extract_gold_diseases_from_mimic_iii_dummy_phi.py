import itertools
import logging
import pathlib
import re
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import nltk
import pandas as pd
from tqdm import tqdm

from kart.src.modules.logging.logger import get_stream_handler
from kart.src.modules.privacy.utils.path import get_repo_dir
from kart.src.modules.pymetamap import pymetamap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = get_stream_handler()
file_handler = logging.FileHandler(f"{pathlib.Path(__file__).stem}.log")
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def main():
    mm = get_metamap_instance()
    for mode in ("hospital", "shadow"):
        for code in ("c1p2", "c1p1", "c1p0", "c0p2", "c0p1", "c0p0"):
            input_path = (
                get_repo_dir()
                / f"corpus/full_name_mentions_{mode.lower()}_{code.lower()}.tsv"
            )
            output_path = (
                get_repo_dir()
                / f"corpus/gold_disease_names_{mode.lower()}_{code.lower()}.tsv"
            )

            logger.info(f"Loading {input_path} ...")

            try:
                df = pd.read_csv(input_path, sep="\t")
            except FileNotFoundError:
                logger.error(
                    f"File does not exist ({input_path}): "
                    + "Please run extract_full_name_mentions_from_mimic_iii_dummy_phi.py first"
                )
                return None

            full_name_mentions = df["full_name_mention"].values
            texts = [text.replace("\n", "") for text in full_name_mentions]
            target_semantic_types = ["dsyn", "mobd", "neop"]
            option = {"restrict_to_sts": target_semantic_types}

            logger.info(
                "Extracting UMLS concepts using MetaMap "
                + f"(target semantic types = {', '.join(target_semantic_types)}) ..."
            )
            concepts: List[List[Dict]] = extract_umls_concepts(texts, mm, option)
            logger.info("Done!")

            chained_concepts: List[Dict] = list(itertools.chain(*concepts))
            if chained_concepts:
                logger.info(f"Sample: {chained_concepts[0]}")

            logger.info("Retrieving superficial forms from full name mentions ...")
            subject_ids = df["SUBJECT_ID"].values
            patient_ages = df["patient_age"].values
            patient_full_names = df["patient_full_name"].values
            diseases = []

            for i in tqdm(range(len(full_name_mentions))):
                for entity in concepts[i]:
                    superficials = entity["superficials"]
                    concept: pymetamap.Concept.ConceptMMI = entity["concept"]

                    dict_values = [
                        ("document_id", i),
                        ("subject_id", subject_ids[i]),
                        ("patient_full_name", " ".join(eval(patient_full_names[i]))),
                        ("patiant_age", patient_ages[i]),
                        ("superficials", ", ".join(superficials)),
                        ("semtypes", concept.semtypes),
                        ("score", concept.score),
                        ("cui", concept.cui),
                        ("perferred_name", concept.preferred_name),
                    ]
                    disease = OrderedDict(dict_values)
                    diseases.append(disease)

            df_diseases = pd.DataFrame(diseases)
            df_diseases.to_csv(output_path, index=False, sep="\t")
            logger.info(f"Disease name saved to {output_path}")


def get_metamap_instance() -> pymetamap.SubprocessBackend:
    mm = pymetamap.MetaMap.get_instance(
        "/home/nakamura/metamap/public_mm/bin/metamap20",
    )
    return mm


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
    entities = convert_concepts_to_superficial_concept_pairs(concepts, sentences)
    entities = drop_invalid_concepts(entities)
    return entities


def text_to_sentences(text: str) -> List[str]:
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentences = tokenizer.sentences_from_text(text)
    return sentences


def drop_invalid_concepts(superficial_concept_pairs: List[Dict]) -> List[Dict]:
    """
    MetaMap is sometimes too sensitive and recognizes non-disease spans as diseases.
    This function gets rid of such excessive entries with rule based approach.
    """
    result = []
    for pair in superficial_concept_pairs:
        superficials = pair["superficials"]
        non_disease_superficials = (
            "was",
            "IV",
            "arms",
            "call",
            "NS",
            "NC",
            "a, o",
            "this",
        )
        if set(superficials) & set(non_disease_superficials):
            continue
        else:
            result.append(pair)
    return result


def convert_concepts_to_superficial_concept_pairs(
    concepts: List[Union[pymetamap.Concept.ConceptMMI, pymetamap.Concept.ConceptAA]],
    sentences: List[str],
) -> List[Dict]:
    """
    Pymetamap does not return directly spans corresponding to each UMLS concept.
    This function thus extracts such spans and returns pairs of UMLS concepts and spans.
    """
    entities = []

    for concept in concepts:
        if type(concept) is pymetamap.Concept.ConceptMMI:
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
        elif type(concept) is pymetamap.Concept.ConceptAA:
            continue
        else:
            continue
    return entities


if __name__ == "__main__":
    main()

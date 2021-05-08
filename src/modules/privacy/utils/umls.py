import re
from typing import Dict, List, Tuple, Union

import nltk
from tqdm import tqdm

from kart.src.modules.pymetamap import pymetamap


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

    try:
        concepts, error = mm.extract_concepts(
            sentences, range(len(sentences)), **option
        )
    except IndexError:
        # pymetamap raises IndexError when invalid characters exist in the text
        concepts = []

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

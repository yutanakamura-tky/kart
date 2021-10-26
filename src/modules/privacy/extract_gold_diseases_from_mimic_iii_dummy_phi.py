# isort: skip_file
# (To prevent conflict of isort and black)
import itertools
import logging
import pathlib
from collections import OrderedDict
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from kart.src.modules.logging.logger import get_stream_handler
from kart.src.modules.privacy.utils.path import get_repo_dir
from kart.src.modules.privacy.utils.umls import (
    extract_umls_concepts,
    get_metamap_instance,
)
from kart.src.modules.pymetamap import pymetamap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = get_stream_handler()
file_handler = logging.FileHandler(f"{pathlib.Path(__file__).stem}.log")
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def main():
    logger.info("=" * 40)
    mm = get_metamap_instance()
    for mode in ("hospital", "shadow"):
        for code in ("c1p2", "c1p1", "c1p0", "c0p2", "c0p1", "c0p0"):
            input_path = (
                get_repo_dir()
                / f"corpus/full_name_mentions_{mode.lower()}_{code.lower()}.tsv"
            )
            disease_output_path = (
                get_repo_dir()
                / f"corpus/gold_disease_names_{mode.lower()}_{code.lower()}.tsv"
            )
            full_name_output_path = (
                get_repo_dir()
                / f"corpus/gold_full_names_{mode.lower()}_{code.lower()}.tsv"
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
                        ("patient_full_name", patient_full_names[i]),
                        ("patiant_age", patient_ages[i]),
                        ("superficials", ", ".join(superficials)),
                        ("semtypes", concept.semtypes),
                        ("score", concept.score),
                        ("cui", concept.cui),
                        ("preferred_name", concept.preferred_name),
                        ("full_name_mention", full_name_mentions[i]),
                    ]
                    disease = OrderedDict(dict_values)
                    diseases.append(disease)

            df_diseases = pd.DataFrame(diseases)
            df_diseases.to_csv(disease_output_path, index=False, sep="\t")
            logger.info(f"Disease names saved to {disease_output_path}")

            df_full_name = df_diseases.drop_duplicates(subset="patient_full_name")

            if len(df_full_name) > 0:
                df_full_name = df_full_name.loc[
                    :,
                    [
                        "document_id",
                        "subject_id",
                        "patient_full_name",
                        "full_name_mention",
                    ],
                ]
            df_full_name.to_csv(full_name_output_path, index=False, sep="\t")
            logger.info(f"Gold full names saved to {full_name_output_path}")


if __name__ == "__main__":
    main()

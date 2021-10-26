# isort: skip_file
# (To prevent conflict of isort and black)
import argparse
import logging
import pandas as pd
import pathlib
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
from kart.src.modules.privacy.utils.namebook import PopularNameBook
from kart.src.modules.privacy.utils.umls import (
    get_metamap_instance,
    extract_umls_concepts,
)
from kart.src.modules.logging.logger import get_stream_handler
from kart.src.modules.privacy.utils.path import get_repo_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = get_stream_handler()
file_handler = logging.FileHandler(f"{pathlib.Path(__file__).stem}.log")
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def main():
    args = get_args()
    input_path = pathlib.Path(args.input_path)
    output_path = input_path.parent / ("pred_info_" + input_path.stem + ".tsv")

    with open(input_path) as f:
        samples = f.readlines()

    df_gold = pd.read_csv(
        get_repo_dir() / "corpus/gold_disease_names_hospital_c0p2.tsv", sep="\t"
    )

    gold_info = [
        tuple(record)
        for record in df_gold.loc[
            :, ["patient_full_name", "cui", "perferred_name"]
        ].to_dict(orient="split")["data"]
    ]

    gold_columns = ["patient_full_name", "cui", "preferred_name"]
    pred_columns = [
        "patient_full_name",
        "cui",
        "preferred_name",
        "sample_index",
        "is_valid_full_name",
    ]
    df_gold_info = pd.DataFrame(gold_info, columns=gold_columns)
    df_gold_info.to_csv(
        get_repo_dir() / "src/modules/privacy/gold_info_hospital_c0p2.tsv",
        sep="\t",
        index=False,
    )

    pred_info = nlg_samples_to_pred_info(
        samples, skip_invalid_full_names=args.skip_invalid_full_names
    )
    df_pred_info = pd.DataFrame(pred_info, columns=pred_columns)
    df_pred_info.to_csv(output_path, sep="\t", index=False)


def select_samples_with_valid_full_names(
    samples: List[str], skip_invalid_full_names: bool = True
) -> Tuple[List[str], List[int], List[bool]]:
    namebook = PopularNameBook()

    pred_full_names = []
    sample_indices = []
    full_name_validities = []

    for i, sample in tqdm(enumerate(samples)):
        full_name_is_valid = False
        match_obj = re.match(r"^\[CLS\] (.+?) is a.+$", sample)

        if not match_obj:
            continue

        pred_full_name = match_obj.group(1)
        pred_full_name_parts = pred_full_name.split()

        if len(pred_full_name_parts) == 2:
            first_name, last_name = pred_full_name_parts

            if (first_name.title() in namebook.first_names_in_vocab) and (
                last_name.title() in namebook.last_names_in_vocab
            ):
                full_name_is_valid = True

        if full_name_is_valid or (not skip_invalid_full_names):
            pred_full_names.append(
                " ".join([part.title() for part in pred_full_name_parts])
            )
            sample_indices.append(i)
            full_name_validities.append(full_name_is_valid)

    return pred_full_names, sample_indices, full_name_validities


def extract_pred_names_and_diseases(
    samples: List[str],
    names: List[str],
    index: List[int],
    full_name_validities: List[bool],
):
    mm = get_metamap_instance()
    target_semantic_types = ["dsyn", "mobd", "neop"]
    option = {"restrict_to_sts": target_semantic_types}

    filtered_samples = [samples[i] for i in index]
    concepts: List[List[Dict]] = extract_umls_concepts(filtered_samples, mm, option)

    raw_pred_info = []

    for concepts_for_one_sample, name, ix, validity in tqdm(
        zip(concepts, names, index, full_name_validities)
    ):
        for disease in concepts_for_one_sample:
            raw_pred_info.append(
                {
                    "pred_name": name,
                    "pred_disease": disease,
                    "sample_index": ix,
                    "is_valid_full_name": validity,
                }
            )

    return raw_pred_info


def nlg_samples_to_pred_info(
    samples: List[str], skip_invalid_full_names: bool = True
) -> List[Tuple]:
    names, index, full_name_validities = select_samples_with_valid_full_names(
        samples, skip_invalid_full_names=skip_invalid_full_names
    )
    raw_pred_info = extract_pred_names_and_diseases(
        samples, names, index, full_name_validities
    )

    pred_info = [
        (
            info["pred_name"],
            info["pred_disease"]["concept"].cui,
            info["pred_disease"]["concept"].preferred_name,
            info["sample_index"],
            info["is_valid_full_name"],
        )
        for info in raw_pred_info
    ]

    return pred_info


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--skip-invalid-full-names", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

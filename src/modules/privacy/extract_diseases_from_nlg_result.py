# isort: skip_file
# (To prevent conflict of isort and black)
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
    RESULT_DIR = get_repo_dir() / "src/modules/privacy"

    PATH_F_KA = (
        RESULT_DIR / "generation_result_model_c0p2_"
        + "iter_1000_batchsize_16_sequential_always_temp_1.0_topk_100_burnin_1000_len_128_"
        + "fullname_known_corpus_unused_no_anonymization.txt_finetuned.txt"
    )

    PATH_F_KAR = (
        RESULT_DIR / "generation_result_model_c0p2_"
        + "iter_1000_batchsize_16_sequential_always_temp_1.0_topk_100_burnin_1000_len_128_"
        + "fullname_unknown_corpus_used_no_anonymization.txt_finetuned.txt"
    )

    with open(PATH_F_KA) as f:
        samples_f_ka = f.readlines()[:5000]

    with open(PATH_F_KAR) as f:
        samples_f_kar = f.readlines()[:5000]

    df_gold = pd.read_csv(
        "/home/nakamura/kart/corpus/gold_disease_names_hospital_c0p2.tsv", sep="\t"
    )

    gold_info = [
        tuple(record)
        for record in df_gold.loc[
            :, ["patient_full_name", "cui", "perferred_name"]
        ].to_dict(orient="split")["data"]
    ]

    columns = ["patient_full_name", "cui", "preferred_name"]
    df_gold_info = pd.DataFrame(gold_info, columns=columns)
    df_gold_info.to_csv(RESULT_DIR / "gold_info_hospital_c0p2.tsv", sep="\t")

    pred_info_f_kar = nlg_samples_to_pred_info(samples_f_kar)
    pred_info_f_ka = nlg_samples_to_pred_info(samples_f_ka)

    df_pred_info_f_kar = pd.DataFrame(pred_info_f_kar, columns=columns)
    df_pred_info_f_kar.to_csv(
        RESULT_DIR / "pred_info_hospital_c0p2_finetuned_k_a_r.tsv", sep="\t"
    )

    df_pred_info_f_ka = pd.DataFrame(pred_info_f_ka, columns=columns)
    df_pred_info_f_ka.to_csv(
        RESULT_DIR / "pred_info_hospital_c0p2_finetuned_k_a.tsv", sep="\t"
    )


def select_samples_with_valid_full_names(
    samples: List[str],
) -> Tuple[List[str], List[int]]:
    namebook = PopularNameBook()

    pred_full_names = []
    using_index = []

    for i, sample in tqdm(enumerate(samples)):
        match_obj = re.match(r"^\[CLS\] (.+?) is a.+$", sample)

        if not match_obj:
            continue

        pred_full_name = match_obj.group(1)
        pred_full_name_parts = pred_full_name.split()

        if len(pred_full_name_parts) != 2:
            continue

        first_name, last_name = pred_full_name_parts

        if first_name.title() in namebook.first_names_in_vocab:
            if last_name.title() in namebook.last_names_in_vocab:
                using_index.append(i)
                pred_full_names.append(f"{first_name.title()} {last_name.title()}")

    return pred_full_names, using_index


def extract_pred_names_and_diseases(samples, names, index):
    mm = get_metamap_instance()
    target_semantic_types = ["dsyn", "mobd", "neop"]
    option = {"restrict_to_sts": target_semantic_types}

    raw_pred_info = []

    for name, ix in tqdm(zip(names, index)):
        pred_diseases: List[List[Dict]] = extract_umls_concepts(
            [samples[ix]], mm, option
        )[0]
        for disease in pred_diseases:
            raw_pred_info.append(
                {"pred_name": name, "pred_disease": disease, "sample_index": ix}
            )

    return raw_pred_info


def nlg_samples_to_pred_info(samples):
    names, index = select_samples_with_valid_full_names(samples)
    raw_pred_info = extract_pred_names_and_diseases(samples, names, index)

    pred_info = [
        (
            info["pred_name"],
            info["pred_disease"]["concept"].cui,
            info["pred_disease"]["concept"].preferred_name,
        )
        for info in raw_pred_info
    ]

    return pred_info


if __name__ == "__main__":
    main()

import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm


def main():
    args = get_args()
    df_all = pd.read_csv(args.input_path, quoting=0, low_memory=False)
    save_dir = pathlib.Path(args.input_path).parent

    # Hospital/Shadow split

    RANDOM_STATE = 42
    df_hospital, df_shadow = train_test_split(
        df_all, train_size=0.5, random_state=RANDOM_STATE
    )
    print("Split MIMIC-III-dummy-PHI into hospital & shadow halves")

    out_path_hospital = save_dir / "MIMIC_III_DUMMY_PHI_HOSPITAL.csv"
    out_path_shadow = save_dir / "MIMIC_III_DUMMY_PHI_SHADOW.csv"

    # Add indices

    df_hospital = add_subset_columns(df_hospital)
    df_hospital.to_csv(out_path_hospital, index=False)
    print(f"Saved hospital half: -> {out_path_hospital} ...")

    df_shadow = add_subset_columns(df_shadow)
    df_shadow.to_csv(out_path_shadow, index=False)
    print(f"Saved shadow half: -> {out_path_shadow} ...")


def add_subset_columns(df):
    tqdm.pandas()

    RANDOM_STATE = 42

    def index_to_bool(indices):
        return np.isin(df.index, indices)

    # 1M corpus
    strap_1M = np.array(shuffle(range(len(df)), random_state=RANDOM_STATE))
    strap_1M_train = strap_1M[:1000000]
    strap_1M_test = strap_1M[1000000:]

    index_train = df.index[strap_1M_train]
    index_test = df.index[strap_1M_test]

    print("Train 1M / Test split ...")
    df["in_train_1M"] = index_to_bool(index_train)
    print('Added column "in_train_1M"')
    df["in_test_1M"] = index_to_bool(index_test)
    print('Added column "in_test_1M"')

    assert (df["in_train_1M"] & df["in_test_1M"]).sum() == 0
    assert df["in_train_1M"].sum() == 1000000

    # Less category diversity
    print(
        "Extracting corpus with less category diversity (Physician notes & Discharge summary only) ..."
    )
    category_filtering_query = 'CATEGORY=="Physician "|CATEGORY=="Discharge summary"'

    strap_100k_less_c_div = np.array(
        shuffle(
            range(len(df.query(category_filtering_query))),
            random_state=RANDOM_STATE,
        )
    )

    index_train_100k_less_c_div = df.query(category_filtering_query).index[
        strap_100k_less_c_div[:100000]
    ]
    index_test_100k_less_c_div = df.query(category_filtering_query).index[
        strap_100k_less_c_div[100000:]
    ]

    print("Train 100k / Test split ...")

    df["in_train_100k_less_c_div"] = index_to_bool(index_train_100k_less_c_div)
    print('Added columns "in_train_100k_less_c_div"')

    df["in_test_100k_less_c_div"] = index_to_bool(index_test_100k_less_c_div)
    print('Added columns "in_test_100k_less_c_div"')

    assert (df["in_train_100k_less_c_div"] & df["in_test_100k_less_c_div"]).sum() == 0

    assert df["in_train_100k_less_c_div"].sum() == 100000
    assert df["in_train_100k_less_c_div"].sum() + df[
        "in_test_100k_less_c_div"
    ].sum() == len(strap_100k_less_c_div)

    assert set(df.query("in_train_100k_less_c_div")["CATEGORY"].values) == {
        "Discharge summary",
        "Physician ",
    }
    assert set(df.query("in_test_100k_less_c_div")["CATEGORY"].values) == {
        "Discharge summary",
        "Physician ",
    }

    # Less patient diversity

    def cut_off_rare_patients(df, desired_n_docs):
        docs_per_patient = (
            df.groupby("SUBJECT_ID").count()["ROW_ID"].sort_values().values
        )
        n_docs_init = docs_per_patient.sum()
        n_patients_init = len(docs_per_patient)
        n_docs_prev = n_docs_init
        n_patients_prev = n_patients_init

        thresh = 0

        while True:
            n_docs = docs_per_patient[docs_per_patient >= thresh].sum()
            n_patients = (docs_per_patient >= thresh).sum()
            if n_docs <= desired_n_docs:
                break
            else:
                thresh += 1
                n_docs_prev = n_docs
                n_patients_prev = n_patients
                continue

        thresh = thresh - 1

        result = {
            "thresh": thresh,
            "n_patients": n_patients_prev,
            "pct_patients": n_patients_prev / n_patients_init * 100,
            "n_docs": n_docs_prev,
            "pct_docs": n_docs_prev / n_docs_init * 100,
        }

        target_patient_id = (
            df.groupby("SUBJECT_ID")
            .count()["ROW_ID"][
                df.groupby("SUBJECT_ID").count()["ROW_ID"].values >= thresh
            ]
            .index
        )

        target_index = df[
            df["SUBJECT_ID"].progress_apply(lambda x: x in target_patient_id)
        ].index

        strap = np.array(shuffle(range(len(target_index)), random_state=RANDOM_STATE))
        train_index = target_index[strap[:desired_n_docs]]
        test_index = target_index[strap[desired_n_docs:]]

        result["train_index"] = train_index
        result["test_index"] = test_index

        return result

    print("Extracting corpus with less patient diversity ...")
    index_train_100k_less_p_div = cut_off_rare_patients(df, 100000)["train_index"]
    index_test_100k_less_p_div = cut_off_rare_patients(df, 100000)["test_index"]

    print("Train 100k / Test split ...")
    df["in_train_100k_less_p_div"] = index_to_bool(index_train_100k_less_p_div)
    print('Added columns "in_train_100k_less_p_div"')

    df["in_test_100k_less_p_div"] = index_to_bool(index_test_100k_less_p_div)
    print('Added columns "in_test_100k_less_p_div"')

    assert (df["in_train_100k_less_p_div"] != df["in_train_100k_less_c_div"]).any()
    assert (df["in_test_100k_less_p_div"] != df["in_test_100k_less_c_div"]).any()

    assert (df["in_train_100k_less_p_div"] & df["in_test_100k_less_p_div"]).sum() == 0

    assert df["in_train_100k_less_p_div"].sum() == 100000

    # Least patient diversity
    print("Extracting corpus with least patient diversity ...")
    index_train_10k_least_p_div = cut_off_rare_patients(df, 10000)["train_index"]
    index_test_10k_least_p_div = cut_off_rare_patients(df, 10000)["test_index"]

    print("Train 10k / Test split ...")
    df["in_train_10k_least_p_div"] = index_to_bool(index_train_10k_least_p_div)
    print('Added columns "in_train_10k_least_p_div"')

    df["in_test_10k_least_p_div"] = index_to_bool(index_test_10k_least_p_div)
    print('Added columns "in_test_10k_least_p_div"')

    assert (df["in_train_10k_least_p_div"] != df["in_train_100k_less_c_div"]).any()
    assert (df["in_test_10k_least_p_div"] != df["in_test_100k_less_c_div"]).any()

    assert (df["in_train_10k_least_p_div"] != df["in_train_100k_less_p_div"]).any()
    assert (df["in_test_10k_least_p_div"] != df["in_test_100k_less_p_div"]).any()

    assert (df["in_train_10k_least_p_div"] & df["in_test_10k_least_p_div"]).sum() == 0

    assert df["in_train_10k_least_p_div"].sum() == 10000

    # Less category & Less patient diversity
    print("Extracting corpus with less category & less patient diversity ...")

    index_train_10k_less_c_less_p_div = cut_off_rare_patients(
        df.query('CATEGORY=="Discharge summary"|CATEGORY=="Physician "'),
        10000,
    )["train_index"]

    index_test_10k_less_c_less_p_div = cut_off_rare_patients(
        df.query('CATEGORY=="Discharge summary"|CATEGORY=="Physician "'),
        10000,
    )["test_index"]

    print("Train 10k / Test split ...")
    df["in_train_10k_less_c_less_p_div"] = index_to_bool(
        index_train_10k_less_c_less_p_div
    )
    print('Added columns "in_train_10k_less_c_less_p_div"')

    df["in_test_10k_less_c_less_p_div"] = index_to_bool(
        index_test_10k_less_c_less_p_div
    )
    print('Added columns "in_test_10k_less_c_less_p_div"')

    assert (
        df["in_train_10k_less_c_less_p_div"] != df["in_train_100k_less_c_div"]
    ).any()
    assert (df["in_test_10k_less_c_less_p_div"] != df["in_test_100k_less_c_div"]).any()

    assert (
        df["in_train_10k_less_c_less_p_div"] != df["in_train_100k_less_p_div"]
    ).any()
    assert (df["in_test_10k_less_c_less_p_div"] != df["in_test_100k_less_p_div"]).any()

    assert (
        df["in_train_10k_less_c_less_p_div"] != df["in_train_10k_least_p_div"]
    ).any()
    assert (df["in_train_10k_less_c_less_p_div"] != df["in_test_10k_least_p_div"]).any()

    assert (
        df["in_train_10k_less_c_less_p_div"] & df["in_test_10k_less_c_less_p_div"]
    ).sum() == 0

    assert df["in_train_10k_less_c_less_p_div"].sum() == 10000

    # Less category & Least patient diversity
    print("Extracting corpus with less category & least patient diversity ...")

    index_train_1k_less_c_least_p_div = cut_off_rare_patients(
        df.query(category_filtering_query), 1000
    )["train_index"]

    index_test_1k_less_c_least_p_div = cut_off_rare_patients(
        df.query(category_filtering_query), 1000
    )["test_index"]

    print("Train 1k / Test split ...")
    df["in_train_1k_less_c_least_p_div"] = index_to_bool(
        index_train_1k_less_c_least_p_div
    )
    print('Added columns "in_train_1k_less_c_least_p_div"')

    df["in_test_1k_less_c_least_p_div"] = index_to_bool(
        index_test_1k_less_c_least_p_div
    )
    print('Added columns "in_test_1k_less_c_least_p_div"')

    assert (
        df["in_train_1k_less_c_least_p_div"] != df["in_train_100k_less_c_div"]
    ).any()
    assert (df["in_test_1k_less_c_least_p_div"] != df["in_test_100k_less_c_div"]).any()

    assert (
        df["in_train_1k_less_c_least_p_div"] != df["in_train_100k_less_p_div"]
    ).any()
    assert (df["in_test_1k_less_c_least_p_div"] != df["in_test_100k_less_p_div"]).any()

    assert (
        df["in_train_1k_less_c_least_p_div"] != df["in_train_10k_least_p_div"]
    ).any()
    assert (df["in_train_1k_less_c_least_p_div"] != df["in_test_10k_least_p_div"]).any()

    assert (
        df["in_train_1k_less_c_least_p_div"] != df["in_train_10k_less_c_less_p_div"]
    ).any()

    assert df["in_train_1k_less_c_least_p_div"].sum() == 1000

    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

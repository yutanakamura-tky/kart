import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm


def main():
    df = load_mimic_iii_dummy_phi()
    df = add_subset_columns(df)
    # Save
    OUTPUT_PATH = "../../corpus/NOTEEVENTS_WITH_DUMMY_PHI_SAMPLE_SELECTION.csv"
    print(f"Saving to {OUTPUT_PATH} ...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Complete!")


def load_mimic_iii_dummy_phi():
    PATH = "../../corpus/NOTEEVENTS_WITH_DUMMY_PHI.csv"
    print(f"Opening {PATH} ...")
    df = pd.read_csv(PATH, quoting=0)
    return df


def hospital_shadow_split(df, random_state):
    print('Adding columns "in_hospital", "in_shadow" ...')
    indices = np.arange(len(df))
    indices_hospital, indices_shadow = train_test_split(
        indices, train_size=0.5, random_state=random_state
    )
    df["in_hospital"] = np.isin(df.index, indices_hospital)
    df["in_shadow"] = np.isin(df.index, indices_shadow)
    print("Done!")
    return df


def add_subset_columns(df):
    tqdm.pandas()

    RANDOM_STATE = 42

    # Hospital-shadow split
    df = hospital_shadow_split(df, RANDOM_STATE)

    def index_to_bool(indices):
        return np.isin(df.index, indices)

    # 1M corpus
    strap_1M = np.array(
        shuffle(range(len(df.query("in_hospital"))), random_state=RANDOM_STATE)
    )
    strap_1M_train = strap_1M[:1000000]
    strap_1M_test = strap_1M[1000000:]

    index_hospital_train = df.query("in_hospital").index[strap_1M_train]
    index_hospital_test = df.query("in_hospital").index[strap_1M_test]
    index_shadow_train = df.query("in_shadow").index[strap_1M_train]
    index_shadow_test = df.query("in_shadow").index[strap_1M_test]

    print("Train 1M / Test split ...")
    df["in_hospital_train_1M"] = index_to_bool(index_hospital_train)
    print('Added column "in_hospital_train_1M"')
    df["in_hospital_test_1M"] = index_to_bool(index_hospital_test)
    print('Added column "in_hospital_test_1M"')
    df["in_shadow_train_1M"] = index_to_bool(index_shadow_train)
    print('Added column "in_shadow_train_1M"')
    df["in_shadow_test_1M"] = index_to_bool(index_shadow_test)
    print('Added column "in_shadow_test_1M"')

    assert (df["in_hospital_train_1M"] & df["in_hospital_test_1M"]).sum() == 0
    assert (df["in_shadow_train_1M"] & df["in_shadow_test_1M"]).sum() == 0
    assert (df["in_hospital_train_1M"] & df["in_shadow_train_1M"]).sum() == 0
    assert (df["in_hospital_test_1M"] & df["in_shadow_test_1M"]).sum() == 0
    assert (df["in_hospital_train_1M"] & df["in_shadow_test_1M"]).sum() == 0
    assert (df["in_shadow_train_1M"] & df["in_hospital_test_1M"]).sum() == 0

    assert df["in_hospital_train_1M"].sum() == 1000000
    assert df["in_shadow_train_1M"].sum() == 1000000

    # Less category diversity
    print(
        "Extracting corpus with less category diversity (Physician notes & Discharge summary only) ..."
    )
    category_filtering_query = 'CATEGORY=="Physician "|CATEGORY=="Discharge summary"'

    strap_hospital_100k_less_c_div = np.array(
        shuffle(
            range(len(df.query("in_hospital").query(category_filtering_query))),
            random_state=RANDOM_STATE,
        )
    )
    strap_shadow_100k_less_c_div = np.array(
        shuffle(
            range(len(df.query("in_shadow").query(category_filtering_query))),
            random_state=RANDOM_STATE,
        )
    )

    index_hospital_train_100k_less_c_div = (
        df.query("in_hospital")
        .query(category_filtering_query)
        .index[strap_hospital_100k_less_c_div[:100000]]
    )
    index_hospital_test_100k_less_c_div = (
        df.query("in_hospital")
        .query(category_filtering_query)
        .index[strap_hospital_100k_less_c_div[100000:]]
    )
    index_shadow_train_100k_less_c_div = (
        df.query("in_shadow")
        .query(category_filtering_query)
        .index[strap_shadow_100k_less_c_div[:100000]]
    )
    index_shadow_test_100k_less_c_div = (
        df.query("in_shadow")
        .query(category_filtering_query)
        .index[strap_shadow_100k_less_c_div[100000:]]
    )

    print("Train 100k / Test split ...")

    df["in_hospital_train_100k_less_c_div"] = index_to_bool(
        index_hospital_train_100k_less_c_div
    )
    print('Added columns "in_hospital_train_100k_less_c_div"')

    df["in_hospital_test_100k_less_c_div"] = index_to_bool(
        index_hospital_test_100k_less_c_div
    )
    print('Added columns "in_hospital_test_100k_less_c_div"')

    df["in_shadow_train_100k_less_c_div"] = index_to_bool(
        index_shadow_train_100k_less_c_div
    )
    print('Added columns "in_shadow_train_100k_less_c_div"')

    df["in_shadow_test_100k_less_c_div"] = index_to_bool(
        index_shadow_test_100k_less_c_div
    )
    print('Added columns "in_shadow_test_100k_less_c_div"')

    assert (
        df["in_hospital_train_100k_less_c_div"] & df["in_hospital_test_100k_less_c_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_100k_less_c_div"] & df["in_shadow_test_100k_less_c_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_100k_less_c_div"] & df["in_shadow_train_100k_less_c_div"]
    ).sum() == 0
    assert (
        df["in_hospital_test_100k_less_c_div"] & df["in_shadow_test_100k_less_c_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_100k_less_c_div"] & df["in_shadow_test_100k_less_c_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_100k_less_c_div"] & df["in_hospital_test_100k_less_c_div"]
    ).sum() == 0

    assert df["in_hospital_train_100k_less_c_div"].sum() == 100000
    assert df["in_shadow_train_100k_less_c_div"].sum() == 100000
    assert df["in_hospital_train_100k_less_c_div"].sum() + df[
        "in_hospital_test_100k_less_c_div"
    ].sum() == len(strap_hospital_100k_less_c_div)
    assert df["in_shadow_train_100k_less_c_div"].sum() + df[
        "in_shadow_test_100k_less_c_div"
    ].sum() == len(strap_shadow_100k_less_c_div)

    assert set(df.query("in_hospital_train_100k_less_c_div")["CATEGORY"].values) == {
        "Discharge summary",
        "Physician ",
    }
    assert set(df.query("in_hospital_test_100k_less_c_div")["CATEGORY"].values) == {
        "Discharge summary",
        "Physician ",
    }
    assert set(df.query("in_shadow_train_100k_less_c_div")["CATEGORY"].values) == {
        "Discharge summary",
        "Physician ",
    }
    assert set(df.query("in_shadow_test_100k_less_c_div")["CATEGORY"].values) == {
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
    index_hospital_train_100k_less_p_div = cut_off_rare_patients(
        df.query("in_hospital"), 100000
    )["train_index"]
    index_hospital_test_100k_less_p_div = cut_off_rare_patients(
        df.query("in_hospital"), 100000
    )["test_index"]
    index_shadow_train_100k_less_p_div = cut_off_rare_patients(
        df.query("in_shadow"), 100000
    )["train_index"]
    index_shadow_test_100k_less_p_div = cut_off_rare_patients(
        df.query("in_shadow"), 100000
    )["test_index"]

    print("Train 100k / Test split ...")
    df["in_hospital_train_100k_less_p_div"] = index_to_bool(
        index_hospital_train_100k_less_p_div
    )
    print('Added columns "in_hospital_train_100k_less_p_div"')

    df["in_hospital_test_100k_less_p_div"] = index_to_bool(
        index_hospital_test_100k_less_p_div
    )
    print('Added columns "in_hospital_test_100k_less_p_div"')

    df["in_shadow_train_100k_less_p_div"] = index_to_bool(
        index_shadow_train_100k_less_p_div
    )
    print('Added columns "in_shadow_train_100k_less_p_div"')

    df["in_shadow_test_100k_less_p_div"] = index_to_bool(
        index_shadow_test_100k_less_p_div
    )
    print('Added columns "in_shadow_test_100k_less_p_div"')

    assert (
        df["in_hospital_train_100k_less_p_div"]
        != df["in_hospital_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_hospital_test_100k_less_p_div"] != df["in_hospital_test_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_train_100k_less_p_div"] != df["in_shadow_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_test_100k_less_p_div"] != df["in_shadow_test_100k_less_c_div"]
    ).any()

    assert (
        df["in_hospital_train_100k_less_p_div"] & df["in_hospital_test_100k_less_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_100k_less_p_div"] & df["in_shadow_test_100k_less_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_100k_less_p_div"] & df["in_shadow_train_100k_less_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_test_100k_less_p_div"] & df["in_shadow_test_100k_less_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_100k_less_p_div"] & df["in_shadow_test_100k_less_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_100k_less_p_div"] & df["in_hospital_test_100k_less_p_div"]
    ).sum() == 0

    assert df["in_hospital_train_100k_less_p_div"].sum() == 100000
    assert df["in_shadow_train_100k_less_p_div"].sum() == 100000

    # Least patient diversity
    print("Extracting corpus with least patient diversity ...")
    index_hospital_train_10k_least_p_div = cut_off_rare_patients(
        df.query("in_hospital"), 10000
    )["train_index"]
    index_hospital_test_10k_least_p_div = cut_off_rare_patients(
        df.query("in_hospital"), 10000
    )["test_index"]
    index_shadow_train_10k_least_p_div = cut_off_rare_patients(
        df.query("in_shadow"), 10000
    )["train_index"]
    index_shadow_test_10k_least_p_div = cut_off_rare_patients(
        df.query("in_shadow"), 10000
    )["test_index"]

    print("Train 10k / Test split ...")
    df["in_hospital_train_10k_least_p_div"] = index_to_bool(
        index_hospital_train_10k_least_p_div
    )
    print('Added columns "in_hospital_train_10k_least_p_div"')

    df["in_hospital_test_10k_least_p_div"] = index_to_bool(
        index_hospital_test_10k_least_p_div
    )
    print('Added columns "in_hospital_test_10k_least_p_div"')

    df["in_shadow_train_10k_least_p_div"] = index_to_bool(
        index_shadow_train_10k_least_p_div
    )
    print('Added columns "in_shadow_train_10k_least_p_div"')

    df["in_shadow_test_10k_least_p_div"] = index_to_bool(
        index_shadow_test_10k_least_p_div
    )
    print('Added columns "in_shadow_test_10k_least_p_div"')

    assert (
        df["in_hospital_train_10k_least_p_div"]
        != df["in_hospital_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_hospital_test_10k_least_p_div"] != df["in_hospital_test_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_train_10k_least_p_div"] != df["in_shadow_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_test_10k_least_p_div"] != df["in_shadow_test_100k_less_c_div"]
    ).any()

    assert (
        df["in_hospital_train_10k_least_p_div"]
        != df["in_hospital_train_100k_less_p_div"]
    ).any()
    assert (
        df["in_hospital_test_10k_least_p_div"] != df["in_hospital_test_100k_less_p_div"]
    ).any()
    assert (
        df["in_shadow_train_10k_least_p_div"] != df["in_shadow_train_100k_less_p_div"]
    ).any()
    assert (
        df["in_shadow_test_10k_least_p_div"] != df["in_shadow_test_100k_less_p_div"]
    ).any()

    assert (
        df["in_hospital_train_10k_least_p_div"] & df["in_hospital_test_10k_least_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_10k_least_p_div"] & df["in_shadow_test_10k_least_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_10k_least_p_div"] & df["in_shadow_train_10k_least_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_test_10k_least_p_div"] & df["in_shadow_test_10k_least_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_10k_least_p_div"] & df["in_shadow_test_10k_least_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_10k_least_p_div"] & df["in_hospital_test_10k_least_p_div"]
    ).sum() == 0

    assert df["in_hospital_train_10k_least_p_div"].sum() == 10000
    assert df["in_shadow_train_10k_least_p_div"].sum() == 10000

    # Less category & Less patient diversity
    print("Extracting corpus with less category & less patient diversity ...")

    index_hospital_train_10k_less_c_less_p_div = cut_off_rare_patients(
        df.query("in_hospital").query(
            'CATEGORY=="Discharge summary"|CATEGORY=="Physician "'
        ),
        10000,
    )["train_index"]

    index_hospital_test_10k_less_c_less_p_div = cut_off_rare_patients(
        df.query("in_hospital").query(
            'CATEGORY=="Discharge summary"|CATEGORY=="Physician "'
        ),
        10000,
    )["test_index"]
    index_shadow_train_10k_less_c_less_p_div = cut_off_rare_patients(
        df.query("in_shadow").query(
            'CATEGORY=="Discharge summary"|CATEGORY=="Physician "'
        ),
        10000,
    )["train_index"]

    index_shadow_test_10k_less_c_less_p_div = cut_off_rare_patients(
        df.query("in_shadow").query(
            'CATEGORY=="Discharge summary"|CATEGORY=="Physician "'
        ),
        10000,
    )["test_index"]

    print("Train 10k / Test split ...")
    df["in_hospital_train_10k_less_c_less_p_div"] = index_to_bool(
        index_hospital_train_10k_less_c_less_p_div
    )
    print('Added columns "in_hospital_train_10k_less_c_less_p_div"')

    df["in_hospital_test_10k_less_c_less_p_div"] = index_to_bool(
        index_hospital_test_10k_less_c_less_p_div
    )
    print('Added columns "in_hospital_test_10k_less_c_less_p_div"')

    df["in_shadow_train_10k_less_c_less_p_div"] = index_to_bool(
        index_shadow_train_10k_less_c_less_p_div
    )
    print('Added columns "in_shadow_train_10k_less_c_less_p_div"')

    df["in_shadow_test_10k_less_c_less_p_div"] = index_to_bool(
        index_shadow_test_10k_less_c_less_p_div
    )
    print('Added columns "in_shadow_test_10k_less_c_less_p_div"')

    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        != df["in_hospital_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_hospital_test_10k_less_c_less_p_div"]
        != df["in_hospital_test_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_train_10k_less_c_less_p_div"]
        != df["in_shadow_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_test_10k_less_c_less_p_div"]
        != df["in_shadow_test_100k_less_c_div"]
    ).any()

    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        != df["in_hospital_train_100k_less_p_div"]
    ).any()
    assert (
        df["in_hospital_test_10k_less_c_less_p_div"]
        != df["in_hospital_test_100k_less_p_div"]
    ).any()
    assert (
        df["in_shadow_train_10k_less_c_less_p_div"]
        != df["in_shadow_train_100k_less_p_div"]
    ).any()
    assert (
        df["in_shadow_test_10k_less_c_less_p_div"]
        != df["in_shadow_test_100k_less_p_div"]
    ).any()

    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        != df["in_hospital_train_10k_least_p_div"]
    ).any()
    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        != df["in_hospital_test_10k_least_p_div"]
    ).any()
    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        != df["in_shadow_train_10k_least_p_div"]
    ).any()
    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        != df["in_shadow_test_10k_least_p_div"]
    ).any()

    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        & df["in_hospital_test_10k_less_c_less_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_10k_less_c_less_p_div"]
        & df["in_shadow_test_10k_less_c_less_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        & df["in_shadow_train_10k_less_c_less_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_test_10k_less_c_less_p_div"]
        & df["in_shadow_test_10k_less_c_less_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_10k_less_c_less_p_div"]
        & df["in_shadow_test_10k_less_c_less_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_10k_less_c_less_p_div"]
        & df["in_hospital_test_10k_less_c_less_p_div"]
    ).sum() == 0

    assert df["in_hospital_train_10k_less_c_less_p_div"].sum() == 10000
    assert df["in_shadow_train_10k_less_c_less_p_div"].sum() == 10000

    # Less category & Least patient diversity
    print("Extracting corpus with less category & least patient diversity ...")

    index_hospital_train_1k_less_c_least_p_div = cut_off_rare_patients(
        df.query("in_hospital").query(category_filtering_query), 1000
    )["train_index"]

    index_hospital_test_1k_less_c_least_p_div = cut_off_rare_patients(
        df.query("in_hospital").query(category_filtering_query), 1000
    )["test_index"]
    index_shadow_train_1k_less_c_least_p_div = cut_off_rare_patients(
        df.query("in_shadow").query(category_filtering_query), 1000
    )["train_index"]

    index_shadow_test_1k_less_c_least_p_div = cut_off_rare_patients(
        df.query("in_shadow").query(category_filtering_query), 1000
    )["test_index"]

    print("Train 1k / Test split ...")
    df["in_hospital_train_1k_less_c_least_p_div"] = index_to_bool(
        index_hospital_train_1k_less_c_least_p_div
    )
    print('Added columns "in_hospital_train_1k_less_c_least_p_div"')

    df["in_hospital_test_1k_less_c_least_p_div"] = index_to_bool(
        index_hospital_test_1k_less_c_least_p_div
    )
    print('Added columns "in_hospital_test_1k_less_c_least_p_div"')

    df["in_shadow_train_1k_less_c_least_p_div"] = index_to_bool(
        index_shadow_train_1k_less_c_least_p_div
    )
    print('Added columns "in_shadow_train_1k_less_c_least_p_div"')

    df["in_shadow_test_1k_less_c_least_p_div"] = index_to_bool(
        index_shadow_test_1k_less_c_least_p_div
    )
    print('Added columns "in_shadow_test_1k_less_c_least_p_div"')

    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_hospital_test_1k_less_c_least_p_div"]
        != df["in_hospital_test_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_train_1k_less_c_least_p_div"]
        != df["in_shadow_train_100k_less_c_div"]
    ).any()
    assert (
        df["in_shadow_test_1k_less_c_least_p_div"]
        != df["in_shadow_test_100k_less_c_div"]
    ).any()

    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_train_100k_less_p_div"]
    ).any()
    assert (
        df["in_hospital_test_1k_less_c_least_p_div"]
        != df["in_hospital_test_100k_less_p_div"]
    ).any()
    assert (
        df["in_shadow_train_1k_less_c_least_p_div"]
        != df["in_shadow_train_100k_less_p_div"]
    ).any()
    assert (
        df["in_shadow_test_1k_less_c_least_p_div"]
        != df["in_shadow_test_100k_less_p_div"]
    ).any()

    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_train_10k_least_p_div"]
    ).any()
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_test_10k_least_p_div"]
    ).any()
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_shadow_train_10k_least_p_div"]
    ).any()
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_shadow_test_10k_least_p_div"]
    ).any()

    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_train_10k_less_c_less_p_div"]
    ).any()
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_train_10k_less_c_less_p_div"]
    ).any()
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_train_10k_less_c_less_p_div"]
    ).any()
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        != df["in_hospital_train_10k_less_c_less_p_div"]
    ).any()

    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        & df["in_hospital_test_1k_less_c_least_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_1k_less_c_least_p_div"]
        & df["in_shadow_test_1k_less_c_least_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        & df["in_shadow_train_1k_less_c_least_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_test_1k_less_c_least_p_div"]
        & df["in_shadow_test_1k_less_c_least_p_div"]
    ).sum() == 0
    assert (
        df["in_hospital_train_1k_less_c_least_p_div"]
        & df["in_shadow_test_1k_less_c_least_p_div"]
    ).sum() == 0
    assert (
        df["in_shadow_train_1k_less_c_least_p_div"]
        & df["in_hospital_test_1k_less_c_least_p_div"]
    ).sum() == 0

    assert df["in_hospital_train_1k_less_c_least_p_div"].sum() == 1000
    assert df["in_shadow_train_1k_less_c_least_p_div"].sum() == 1000

    return df


if __name__ == "__main__":
    main()

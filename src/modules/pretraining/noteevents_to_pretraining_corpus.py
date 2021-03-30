# -*- coding: utf-8 -*-
#
# This script is a modification of the notebook by Kexin Huang:
# https://github.com/kexinhuang12345/clinicalBERT/blob/master/notebook/pretrain.ipynb

import argparse
import gc
import os
import re
import string
from pathlib import Path

import pandas as pd
from spacy.lang.en import English
from tqdm import tqdm

from modules import add_subset_columns


def main():
    args = get_args()
    make_pretraining_corpus(args.dataset_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="dataset_dir", type=str)
    args = parser.parse_args()
    return args


def make_pretraining_corpus(dataset_dir):
    """
    inputs
    ------
    corpus (str): Set 'hospital', 'shadow' or 'debug' to format corpus.
    """
    tqdm.pandas()

    # STEP 1: load Note datasets
    INPUT_DIR = Path(dataset_dir)
    OUTPUT_DIR = INPUT_DIR / "pretraining_corpus"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    INPUT_PATH = INPUT_DIR / "NOTEEVENTS_WITH_DUMMY_PHI.csv"
    print(f"Loading noteevents from {INPUT_PATH} ...")
    df = add_subset_columns(pd.read_csv(INPUT_PATH, quoting=0, low_memory=False))

    # STEP 2: Preprocessing
    print("Preprocessing ...")
    print("(1/3) D_private Corpus (no anonymization) ...")
    df_hospital = preprocessing(
        df["TEXT_WITH_DUMMY_PHI_HOSPITAL"].loc[df["in_hospital"]], hipaa=False
    )
    print("(2/3) D_shadow Corpus (no anonymization) ...")
    df_shadow = preprocessing(
        df["TEXT_WITH_DUMMY_PHI_SHADOW"].loc[df["in_shadow"]], hipaa=False
    )
    print("(3/3) D_public Corpus (anonymized under the HIPAA Privacy Rule) ...")
    df_hipaa = preprocessing(df["TEXT"], hipaa=True)
    print("Preprocessing complete.")

    # for memory efficiency
    df = df.drop("TEXT", axis=1)
    df = df.drop("TEXT_WITH_DUMMY_PHI_HOSPITAL", axis=1)
    df = df.drop("TEXT_WITH_DUMMY_PHI_SHADOW", axis=1)

    # STEP 3: Split noteevents to Sentences & saving
    print("Converting notes to sentences & Saving to files ...")
    corpus_codes = ["c1p2", "c0p2", "c1p1", "c0p1", "c1p0", "c0p0"]
    nlp = English()  # just the language with no model
    nlp.add_pipe(nlp.create_pipe("sentencizer"))

    print("(1/3) D_private Corpus (no anonymization) ...")
    df_hospital_sentences = df_hospital.progress_apply(lambda x: to_sentence(nlp, x))
    for i, code in enumerate(corpus_codes):
        print(f"({i+1+6*0}/24) Saving D_private without anonymization ({code})")
        save_noteevents_as_pretraining_corpus(
            df_hospital_sentences, df, OUTPUT_DIR, "hospital", code, "no_anonymization"
        )
    del df_hospital, df_hospital_sentences
    gc.collect()

    print("(2/3) D_shadow Corpus (no anonymization) ...")
    df_shadow_sentences = df_shadow.progress_apply(lambda x: to_sentence(nlp, x))
    for i, code in enumerate(corpus_codes):
        print(f"({i+1+6*1}/24) Saving D_shadow without anonymization ({code})")
        save_noteevents_as_pretraining_corpus(
            df_shadow_sentences, df, OUTPUT_DIR, "shadow", code, "no_anonymization"
        )
    del df_shadow, df_shadow_sentences
    gc.collect()

    print("(3/3) D_public Corpus (anonymized under the HIPAA Privacy Rule) ...")
    df_hipaa_sentences = df_hipaa.progress_apply(lambda x: to_sentence(nlp, x))
    for i, code in enumerate(corpus_codes):
        print(
            f"({i*2+1+6*2}/24) Saving D_private anonymized under the HIPAA Privacy Rule ({code})"
        )
        save_noteevents_as_pretraining_corpus(
            df_hipaa_sentences, df, OUTPUT_DIR, "hospital", code, "hipaa"
        )
        print(
            f"({i*2+2+6*2}/24) Saving D_shadow anonymized under the HIPAA Privacy Rule ({code})"
        )
        save_noteevents_as_pretraining_corpus(
            df_hipaa_sentences, df, OUTPUT_DIR, "shadow", code, "hipaa"
        )
    print("Complete!")


def preprocess1(x, hipaa=False):
    y = re.sub(r"\[(.*?)\]", "", x)  # remove de-identified brackets
    y = re.sub(
        r"[0-9]+\.", "", y
    )  # remove 1.2. since the segmenter segments based on this
    y = re.sub(r"dr\.", "doctor", y)
    y = re.sub(r"m\.d\.", "md", y)
    y = re.sub("--|__|==", "", y)

    # remove, digits, spaces
    if hipaa:
        y = re.sub("admission date:", "", y)
        y = re.sub("discharge date:", "", y)
        y = y.translate(str.maketrans("", "", string.digits))

    y = " ".join(y.split())
    return y


def preprocessing(series, hipaa=False):
    series = series.fillna(" ")
    series = series.str.replace("\n", " ")
    series = series.str.replace("\r", " ")
    series = series.apply(str.strip)
    series = series.str.lower()
    series = series.progress_apply(lambda x: preprocess1(x, hipaa))
    return series


def to_sentence(spacy_func, text):
    # This function is practically the same as ClinicalBERT but refactored for acceleration
    doc = spacy_func(text)
    sentences = list(map(lambda x: str(x).strip(), doc.sents))

    i_left = 0
    i_right = 1
    while i_right < len(sentences):
        if len(sentences[i_right]) < 20:
            sentences[i_left] += " " + sentences[i_right]
            sentences[i_right] = ""
        else:
            sentences[i_left] += "\n"
            i_left = i_right
        i_right += 1

    return "".join(sentences)


def save_noteevents_as_pretraining_corpus(
    df_target, df_master, output_dir, corpus_type, corpus_code, anonymization
):
    out_path = get_out_path(output_dir, corpus_type, corpus_code, anonymization)
    df_target_without_unusing_rows = drop_unnecessary_rows(
        df_target, df_master, corpus_type, corpus_code, anonymization
    )
    noteevents_to_txt(df_target_without_unusing_rows, out_path)


def get_target_column(corpus_type, corpus_code):
    """
    Patameters
    ----------
    df_target: pandas.DataFrame
    df_master: pandas.DataFrame
    corpus_type: str ('hospital', 'shadow')
    corpus_code: str ('c1p2', 'c0p2', 'c1p1', 'c0p1', 'c1p0', 'c0p0')

    Returns
    -------
    pandas.DataFrame
    """
    target_column_prefix = f"in_{corpus_type}_train_"
    target_column_suffix_map = {
        "c1p2": "1M",
        "c0p2": "100k_less_c_div",
        "c1p1": "100k_less_p_div",
        "c0p1": "10k_less_c_less_p_div",
        "c1p0": "10k_least_p_div",
        "c0p0": "1k_less_c_least_p_div",
    }
    return target_column_prefix + target_column_suffix_map[corpus_code]


def drop_unnecessary_rows(
    df_target, df_master, corpus_type, corpus_code, anonymization
):
    """
    Patameters
    ----------
    df_target: pandas.DataFrame
    df_master: pandas.DataFrame
    corpus_type: str ('hospital', 'shadow')
    corpus_code: str ('c1p2', 'c0p2', 'c1p1', 'c0p1', 'c1p0', 'c0p0')
    anonymization: str ('hipaa', 'no_anonymization')

    Returns
    -------
    pandas.DataFrame
    """
    target_column = get_target_column(corpus_type, corpus_code)
    if anonymization == "no_anonymization":
        return df_target.loc[df_master.query(f"in_{corpus_type}").loc[:, target_column]]
    elif anonymization == "hipaa":
        return df_target.loc[df_master.loc[:, target_column]]


def get_out_path(output_dir, corpus_type, corpus_code, anonymization):
    """
    Parameters
    ----------
    output_dir: str
    corpus_type: str ('hospital', 'shadow')
    corpus_code: str ('c1p2', 'c0p2', 'c1p1', 'c0p1', 'c1p0', 'c0p0')
    anonymization: str ('hipaa', 'no_anonymization')

    Returns
    -------
    PosixPath
    """

    out_basename = f"pretraining_corpus_{corpus_type}_{corpus_code}_{anonymization}.txt"
    return Path(output_dir) / out_basename


def noteevents_to_txt(df, out_path):
    """
    Parameters
    ----------
    df: pandas.DataFrame
    out_path: str or path-like object
    """
    with open(out_path, "w") as f:
        f.writelines(df.values.tolist())
    print(f"Saved! -> {out_path}")


if __name__ == "__main__":
    main()

import argparse
import re

import pandas as pd
from tqdm import tqdm


def main():
    print("===== 5. Integrating all noteevents into one csv =====")
    args = get_args()
    tqdm.pandas()
    input_mimic_file_path = f"{args.dataset_dir}/NOTEEVENTS.csv"
    output_path = f"{args.dataset_dir}/NOTEEVENTS_WITH_DUMMY_PHI.csv"

    print(f"Loading {input_mimic_file_path} ...")
    df = pd.read_csv(input_mimic_file_path, quoting=0, low_memory=False)
    for corpus_name in args.corpus_names:
        df = concatenate_noteevents_with_dummy_phi(df, args.dataset_dir, corpus_name)
    print("Saving ...")
    df.to_csv(output_path, index=False)
    print(f"Complete! -> {output_path}")


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(dest="dataset_dir")
    parser.add_argument(dest="corpus_names", nargs="+")
    args = parser.parse_args()
    return args


def concatenate_noteevents_with_dummy_phi(df, dataset_dir, corpus_name):
    pseudonymized_noteevents_text_path = (
        f"{dataset_dir}/tmp/dummy_phi/{corpus_name}/noteevents_text_with_dummy_phi.csv"
    )
    print(f"Loading {pseudonymized_noteevents_text_path} ...")
    df_pseudonymized_text = pd.read_csv(
        pseudonymized_noteevents_text_path, header=None, quoting=0
    )
    pseudonymized_text_col_name = f"TEXT_WITH_DUMMY_PHI_{corpus_name.upper()}"
    df[pseudonymized_text_col_name] = df_pseudonymized_text[1]

    r_placeholder = re.compile(r"\[\*\*.+?\*\*\]")

    if "all_placeholders_replaced" not in df.columns:
        df["all_placeholders_replaced"] = df[pseudonymized_text_col_name].map(
            lambda x: not bool(r_placeholder.findall(x))
        )

    # Select records where all placeholders have IDs
    placeholders_without_id = [
        "[** **]",
        "[**Age over 90 **]",
        "[**Attending Info **]",
        "[**CC Contact Info **]",
        "[**Apartment Address(1) **]",
        "[**Street Address(1) **]",
        "[**Street Address(2) **]",
        "[**Company **]",
        "[**Country **]",
        "[**E-mail address **]",
        "[** First Name **]",
        "[**Doctor First Name **]",
        "[**First Name (STitle) **]",
        "[**First Name (Titles) **]",
        "[**First Name11 (Name Pattern1) **]",
        "[**First Name3 (LF) **]",
        "[**First Name4 (NamePattern1) **]",
        "[**First Name5 (NamePattern1) **]",
        "[**First Name7 (NamePattern1) **]",
        "[**First Name8 (NamePattern2) **]",
        "[**First Name9 (NamePattern2) **]",
        "[**Known firstname **]",
        "[**Name10 (NameIs) **]",
        "[**Name6 (MD) **]",
        "[**Dictator Info **]",
        "[**Doctor Last Name (ambig) **]",
        "[**Doctor Last Name **]",
        "[**First Name (STitle) **]",
        "[**Known lastname **]",
        "[**Last Name (LF) **]",
        "[**Last Name (NamePattern1) **]",
        "[**Last Name (NamePattern4) **]",
        "[**Last Name (NamePattern5) **]",
        "[**Last Name (Prefixes) **]",
        "[**Last Name (STitle) **]",
        "[**Last Name (Titles) **]",
        "[**Last Name (ambig) **]",
        "[**Last Name (un) **]",
        "[**Last Name **]",
        "[**Name (STitle) **]",
        "[**Name11 (NameIs) **]",
        "[**Name13 (STitle) **]",
        "[**Name14 (STitle) **]",
        "[**Name7 (MD) **]",
        "[**Name8 (MD) **]",
        "[**Female First Name (ambig) **]",
        "[**Female First Name (un) **]",
        "[**Male First Name (un) **]",
        "[**Name Prefix (Prefixes) **]",
        "[**Initial (NamePattern1) **]",
        "[**Initials (NamePattern4) **]",
        "[**Initials (NamePattern5) **]",
        "[**Name Initial (MD) **]",
        "[**Name Initial (NameIs) **]",
        "[**Name Initial (PRE) **]",
        "[**Name12 (NameIs) **]",
        "[**Hospital **]",
        "[**Hospital1 **]",
        "[**Hospital2 **]",
        "[**Hospital3 **]",
        "[**Hospital4 **]",
        "[**Hospital5 **]",
        "[**Hospital6 **]",
        "[**Clip Number (Radiology) **]",
        "[**Job Number **]",
        "[**MD Number(1) **]",
        "[**MD Number(2) **]",
        "[**MD Number(3) **]",
        "[**MD Number(4) **]",
        "[**Medical Record Number **]",
        "[**Serial Number **]",
        "[**Month (only) **]",
        "[**Day Month **]",
        "[**Day Month Year **]",
        "[**Month Day **]",
        "[**Month Day Year (2) **]",
        "[**Month/Day (1) **]",
        "[**Month/Day (2) **]",
        "[**Month/Day (3) **]",
        "[**Month/Day (4) **]",
        "[**Day Month Year **]",
        "[**Month/Year (2) **]",
        "[**Month/Year 1 **]",
        "[**Year/Month **]",
        "[**Month/Day/Year **]",
        "[**Year/Month/Day **]",
        "[**Telephone/Fax (1) **]",
        "[**Telephone/Fax (2) **]",
        "[**Telephone/Fax (3) **]",
        "[**Telephone/Fax (5) **]",
        "[**Social Security Number **]",
        "[**State **]",
        "[**University/College **]",
        "[**URL **]",
    ]

    if "all_placeholders_have_id" not in df.columns:
        df["all_placeholders_have_id"] = False
        df["all_placeholders_have_id"] = df["TEXT"].progress_map(
            lambda x: all([p not in x for p in placeholders_without_id])
        )

    return df


if __name__ == "__main__":
    main()

import pathlib
import re

import pandas as pd
from tqdm import tqdm


def get_repo_dir() -> pathlib.PosixPath:
    this_dir = pathlib.Path(__file__).parent
    return (this_dir / "../../..").resolve()


def add_full_name_columns(df):
    tqdm.pandas()
    regexp_patient = regexp_for_full_name()

    df["patient_full_name_placeholder"] = df["TEXT"].progress_apply(
        lambda x: re.compile(regexp_patient).findall(x)[0]
        if re.compile(regexp_patient).findall(x)
        else ()
    )
    print("Added column 'patient_full_name_placeholder'")

    mapping_path_hospital = (
        get_repo_dir() / "corpus/tmp/dummy_phi/hospital/surrogate_map.csv"
    )
    mapping_path_shadow = (
        get_repo_dir() / "corpus/tmp/dummy_phi/shadow/surrogate_map.csv"
    )

    print(f"Loading {mapping_path_hospital} ...")
    print(f"Loading {mapping_path_shadow} ...")

    df_map_hospital = pd.read_csv(mapping_path_hospital, header=None, quoting=0)
    df_map_shadow = pd.read_csv(mapping_path_shadow, header=None, quoting=0)
    mapping_hospital = {
        df_map_hospital.iloc[i].loc[0]: df_map_hospital.iloc[i].loc[1]
        for i in tqdm(range(len(df_map_hospital)))
    }
    mapping_shadow = {
        df_map_shadow.iloc[i].loc[0]: df_map_shadow.iloc[i].loc[1]
        for i in tqdm(range(len(df_map_shadow)))
    }

    df["patient_full_name_hospital"] = df[
        "patient_full_name_placeholder"
    ].progress_apply(
        lambda x: (mapping_hospital[x[0]], mapping_hospital[x[1]]) if x else ()
    )
    print("Added column 'patient_full_name_hospital'")

    df["patient_full_name_shadow"] = df["patient_full_name_placeholder"].progress_apply(
        lambda x: (mapping_shadow[x[0]], mapping_shadow[x[1]]) if x else ()
    )
    print("Added column 'patient_full_name_shadow'")

    df["patient_full_name_tfreq"] = df.progress_apply(
        lambda row: len(
            re.compile(
                escape(row["patient_full_name_placeholder"][0])
                + r"\s*"
                + escape(row["patient_full_name_placeholder"][1])
            ).findall(row["TEXT"])
        )
        if row["patient_full_name_placeholder"]
        else 0,
        axis=1,
    )
    print("Added column 'patient_full_name_tfreq'")
    print("Done!")

    return df


def escape(string):
    string = string.replace("[", "\[")  # noqa
    string = string.replace("]", "\]")  # noqa
    string = string.replace("*", "\*")  # noqa
    string = string.replace("(", "\(")  # noqa
    string = string.replace(")", "\)")  # noqa
    return string


def regexp_for_full_name_mention(n_sentences: int) -> str:
    """
    regexp for full name mention beginning with '(full name) is a/an (age) (sex).'

    Parameters
    ----------
    n_sentences: int
        number of sentences to extract including '(full name) is a/an (age) (sex).'

    Return
    ------
    exp: str
    """
    exp = (
        rf'({regexp_for_name("first", True)}\s*{regexp_for_name("last", True)}\s*'
        + r"is\s+an?\s*(\d+|\[\*\*Age over 90 \*\*\]).*?[Yy].*?[Oo]"
        + r"[^.]*." * (n_sentences)
        + r")"
    )
    return exp


def regexp_for_full_name(placeholder_id_required: bool = True) -> str:
    exp = (
        rf'{regexp_for_name("first", placeholder_id_required)}'
        + rf'\s*{regexp_for_name("last", placeholder_id_required)}\s*'
        + r"is\s+an?\s*(\d+|\[\*\*Age over 90 \*\*\]).*?[Yy].*?[Oo].*?"
    )
    return exp


def regexp_for_name(name_type, placeholder_id_required: bool = True) -> str:
    if name_type == "first":
        template = [
            r"\[\*\*Doctor First Name {}?\*\*\]",
            r"\[\*\*Female First Name \(\w+\) {}?\*\*\]",
            r"\[\*\* First Name \*\*\]",
            r"\[\*\*First Name \(S?Titles?\) {}?\*\*\]",
            r"\[\*\*First Name\d+ \(Name Pattern\d\) {}?\*\*\]",
            r"\[\*\*First Name3 \(LF\) {}?\*\*\]",
            r"\[\*\*Known firstname {}?\*\*\]",
            r"\[\*\*Male First Name \(\w+\) {}?\*\*\]",
            r"\[\*\*Name10 \(NameIs\) {}?\*\*\]",
            r"\[\*\*Name6 \(MD\) {}?\*\*\]",
        ]

    elif name_type == "last":
        template = [
            r"\[\*\*Doctor Last Name {}?\*\*\]",
            r"\[\*\*Known lastname {}?\*\*\]",
            r"\[\*\*Last Name\d*? \(\S+?\) {}?\*\*\]",
            r"\[\*\*Last Name {}?\*\*\]",
            r"\[\*\*Name11 \(NameIs\) {}?\*\*\]",
            r"\[\*\*Name1?[345]? \([SP]?Titles?\) {}?\*\*\]",
            r"\[\*\*Name[78] \(MD\) {}?\*\*\]",
            r"\[\*\*Name2? \(NI\) {}?\*\*\]",
        ]

    elif name_type == "prefix":
        template = [r"\[\*\*Name Prefix \(Prefixes\) {}?\*\*\]"]

    elif name_type == "initial":
        template = [
            r"\[\*\*Initials? \(NamePattern\d\) {}?\*\*\]",
            r"\[\*\*Name Initial \(\w+?\) {}?\*\*\]",
            r"\[\*\*Name12 \(\w+?\) {}?\*\*\]",
        ]

    elif name_type == "not_name":
        template = [
            r"\[\*\*Age over 90 {}?\*\*\]",
            r"\[\*\*Apartment Address\(1\) {}?\*\*\]",
            r"\[\*\*April {}?\*\*\]",
            r"\[\*\*A[tu][ A-Za-z]+ {}?\*\*\]",
            r"\[\*\*Date range \([1-3]\) {}?\*\*\]",
            r"\[\*\*D[aei][ A-Za-z()-]+ {}?\*\*\]",
            r"\[\*\*[CEHJOPSTUWY][ A-Za-z()-/]+ {}?\*\*\]",
            r"\[\*\*Hospital[1-6] {}?\*\*\]",
            r"\[\*\*February {}?\*\*\]",
            r"\[\*\*Lo[ A-Za-z()-]+ {}?\*\*\]",
            r"\[\*\*MD Number\([1-4]\) {}?\*\*\]",
            r"\[\*\*March {}?\*\*\]",
            r"\[\*\*May {}?\*\*\]",
            r"\[\*\*M[eo][ A-Za-z()-]+ {}?\*\*\]",
            r"\[\*\*Month Day Year \(2\) {}?\*\*\]",
            r"\[\*\*Month/Day {}?\*\*\]",
            r"\[\*\*Month/Day [1-4()]* {}?\*\*\]",
            r"\[\*\*Month/Day/Year {}?\*\*\]",
            r"\[\*\*Month/Year [1-2()]+ {}?\*\*\]",
            r"\[\*\*N[ou][ A-Za-z()-]+ {}?\*\*\]",
            r"\[\*\*Street Address\([1-2]\) {}?\*\*\]",
            r"\[\*\*Telephone/Fax \([1-5]\) {}?\*\*\]",
            r"\[\*\*Year \([24] digits\) {}?\*\*\]",
            r"\[\*\*[ 0-9/-]+\*\*\]",
        ]

    if placeholder_id_required:
        return "(" + "|".join([t.format(r"\d+") for t in template]) + ")"
    else:
        return "(" + "|".join([t.format(r"\d*") for t in template]) + ")"


def name_placeholder_ptn(name_type, placeholder_id_required=True):
    if name_type == "first":
        template = [
            r"\[\*\*Doctor First Name {}?\*\*\]",
            r"\[\*\*Female First Name \(\w+\) {}?\*\*\]",
            r"\[\*\* First Name \*\*\]",
            r"\[\*\*First Name \(S?Titles?\) {}?\*\*\]",
            r"\[\*\*First Name\d+ \(Name Pattern\d\) {}?\*\*\]",
            r"\[\*\*First Name3 \(LF\) {}?\*\*\]",
            r"\[\*\*Known firstname {}?\*\*\]",
            r"\[\*\*Male First Name \(\w+\) {}?\*\*\]",
            r"\[\*\*Name10 \(NameIs\) {}?\*\*\]",
            r"\[\*\*Name6 \(MD\) {}?\*\*\]",
        ]

    elif name_type == "last":
        template = [
            r"\[\*\*Doctor Last Name {}?\*\*\]",
            r"\[\*\*Known lastname {}?\*\*\]",
            r"\[\*\*Last Name\d*? \(\S+?\) {}?\*\*\]",
            r"\[\*\*Last Name {}?\*\*\]",
            r"\[\*\*Name11 \(NameIs\) {}?\*\*\]",
            r"\[\*\*Name1?[345]? \([SP]?Titles?\) {}?\*\*\]",
            r"\[\*\*Name[78] \(MD\) {}?\*\*\]",
            r"\[\*\*Name2? \(NI\) {}?\*\*\]",
        ]

    elif name_type == "prefix":
        template = [r"\[\*\*Name Prefix \(Prefixes\) {}?\*\*\]"]

    elif name_type == "initial":
        template = [
            r"\[\*\*Initials? \(NamePattern\d\) \d*?\*\*\]",
            r"\[\*\*Name Initial \(\w+?\) \d*?\*\*\]",
            r"\[\*\*Name12 \(\w+?\) \d*?\*\*\]",
        ]

    if placeholder_id_required:
        return [t.format(r"\d+") for t in template]
    else:
        return [t.format(r"\d*") for t in template]

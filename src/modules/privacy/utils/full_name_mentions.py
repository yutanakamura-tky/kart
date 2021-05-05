import re
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

from kart.src.modules.privacy.utils.path import get_repo_dir


def load_placeholder_info_mapping(mode: str) -> Dict[str, str]:
    mapping_path = get_repo_dir() / f"corpus/tmp/dummy_phi/{mode}/surrogate_map.csv"
    print(f"Loading {mapping_path} ...")
    df_map = pd.read_csv(
        mapping_path, header=None, quoting=0, names=["placeholder", "dummy_phi"]
    )
    mapping = {
        df_map.iloc[i].loc["placeholder"]: df_map.iloc[i].loc["dummy_phi"]
        for i in tqdm(range(len(df_map)))
    }
    return mapping


def add_full_name_columns(
    df: pd.DataFrame, placeholder_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    mode: 'hospital' or 'shadow'
    """
    tqdm.pandas()

    patient_info = df["TEXT"].progress_apply(
        lambda x: extract_patient_info_from_full_name_mention(x, n_sentences=5)
    )

    df["patient_full_name_placeholder"] = patient_info.progress_apply(
        lambda x: (x["first_name_placeholder"], x["last_name_placeholder"])
        if x["first_name_placeholder"] is not None
        else ()
    )
    print("Added column 'patient_full_name_placeholder'")

    for key_name in ("full_name_mention", "patient_age"):
        df[key_name] = patient_info.progress_apply(lambda x: x[key_name])
        print(f"Added column '{key_name}'")

    df["patient_full_name"] = patient_info.progress_apply(
        lambda x: " ".join(
            [
                placeholder_mapping.get(x["first_name_placeholder"], ""),
                placeholder_mapping.get(x["last_name_placeholder"], ""),
            ]
        )
    )
    print("Added column 'patient_full_name'")

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


def extract_patient_info_from_full_name_mention(
    text: str, n_sentences: int
) -> Dict[str, Optional[str]]:
    r = re.compile(regexp_for_full_name_mention(n_sentences=n_sentences))
    matches = r.findall(text)
    result = {
        "full_name_mention": matches[0][0] if matches else None,
        "first_name_placeholder": matches[0][1] if matches else None,
        "last_name_placeholder": matches[0][2] if matches else None,
        "patient_age": matches[0][3] if matches else None,
    }
    return result


def regexp_for_full_name_mention(
    n_sentences: int = 5, placeholder_id_required: bool = True
) -> str:
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
        r"("
        + regexp_for_patient_full_name(placeholder_id_required=placeholder_id_required)
        + r"[^.]*." * (n_sentences)
        + r")"
    )
    return exp


def regexp_for_patient_full_name(placeholder_id_required: bool = True) -> str:
    exp = (
        rf'{regexp_for_name("first", placeholder_id_required)}'
        + rf'\s*{regexp_for_name("last", placeholder_id_required)}\s*'
        + r"is\s+an?\s*(\d+|\[\*\*Age over 90 \*\*\]).*?[Yy].*?[Oo].*?"
    )
    return exp


def regexp_for_name(name_type: str, placeholder_id_required: bool = True) -> str:
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

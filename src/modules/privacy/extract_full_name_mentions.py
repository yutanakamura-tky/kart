import pandas as pd

from kart.src.modules.privacy.utils.full_name_mentions import add_full_name_columns
from kart.src.modules.privacy.utils.path import get_repo_dir


def main():
    df_hospital = load_mimic_iii_dummy_phi("hospital")
    df_shadow = load_mimic_iii_dummy_phi("shadow")

    df_hospital = df_hospital.drop("TEXT_WITH_DUMMY_PHI_SHADOW", axis="columns")
    df_shadow = df_shadow.drop("TEXT_WITH_DUMMY_PHI_HOSPITAL", axis="columns")

    df_hospital = add_full_name_columns(df_hospital, "hospital")
    df_shadow = add_full_name_columns(df_shadow, "shadow")

    code_to_column_suffix = {
        "c1p2": "1M",
        "c0p2": "100k_less_c_div",
        "c1p1": "100k_less_p_div",
        "c1p0": "10k_least_p_div",
        "c0p1": "10k_less_c_less_p_div",
        "c0p0": "1k_less_c_least_p_div",
    }

    for code in ("c1p2", "c1p1", "c1p0", "c0p2", "c0p1", "c0p0"):
        suffix = code_to_column_suffix[code]
        query = f"in_train_{suffix}|in_test_{suffix}"
        save_csv(df_hospital.query(query), mode="hospital", code=code)
        save_csv(df_shadow.query(query), mode="shadow", code=code)


def load_mimic_iii_dummy_phi(mode: str) -> pd.DataFrame:
    """
    mode: 'hospital' or 'shadow'
    """
    corpus_dir = get_repo_dir() / "corpus"
    path = corpus_dir / f"MIMIC_III_DUMMY_PHI_{mode.upper()}.csv"

    print(f"Loading {path} ...")
    df = pd.read_csv(path, quoting=0)
    print("Loaded!")
    return df


def save_csv(df: pd.DataFrame, mode: str, code: str):
    """
    mode:Literal['hospital','shadow']
    code:Literal['c1p2','c1p1','c1p0','c0p2','c0p1','c0p0']
    """
    save_dir = get_repo_dir() / "corpus"
    save_basename = f"MIMIC_III_DUMMY_PHI_{mode.upper()}_{code.upper()}.csv"
    save_path = save_dir / save_basename
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}!")


if __name__ == "__main__":
    main()

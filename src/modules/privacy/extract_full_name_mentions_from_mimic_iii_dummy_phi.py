# isort: skip_file
# (to resolve conflict of isort & black)
import logging
import pathlib

import pandas as pd

from kart.src.modules.logging.logger import get_stream_handler
from kart.src.modules.privacy.utils.full_name_mentions import (
    add_full_name_columns,
    load_placeholder_info_mapping,
)
from kart.src.modules.privacy.utils.namebook import PopularNameBook
from kart.src.modules.privacy.utils.path import get_repo_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = get_stream_handler()
file_handler = logging.FileHandler(f"{pathlib.Path(__file__).stem}.log")
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def main():
    for mode in ("hospital", "shadow"):
        mapping = load_placeholder_info_mapping(mode)

        for code in ("c1p2", "c1p1", "c1p0", "c0p2", "c0p1", "c0p0"):
            df = load_mimic_iii_dummy_phi(mode, code, logger=logger)
            logger.info(f"Loaded DataFrame (n={len(df)})")

            df = add_full_name_columns(df, mapping)

            df = df.query("patient_full_name_tfreq > 0")
            logger.info(f"Extracted records with full name mentions (n={len(df)})")

            df = df.drop_duplicates(subset="full_name_mention")
            logger.info(
                f"Extracted records with unique full name mentions (n={len(df)})"
            )

            j = InVocabJudger()

            bools = df["patient_full_name"].apply(
                lambda x: j.full_name_is_in_vocab(*x.split())
            )
            df = df.loc[bools, :]

            logger.info(
                "Extracted records "
                + "in which first & last names are expressed as one token "
                + f"(n={len(df)})"
            )

            df = df.drop_duplicates(subset="SUBJECT_ID")
            logger.info("Picked up one record per one subject " + f"(n={len(df)})")

            # Save full name mention info as full_name_mentions_{mode}_{code}.tsv
            save_path = get_save_path(mode, code)
            df.to_csv(save_path, index=False, sep="\t")
            logger.info(f"Saved full name mentions to {save_path}!")


class InVocabJudger:
    def __init__(self):
        self.namebook = PopularNameBook()

    def full_name_is_in_vocab(self, first_name: str, last_name: str) -> bool:
        first_name_is_in_vocab = first_name in self.namebook.first_names_in_vocab
        last_name_is_in_vocab = last_name in self.namebook.last_names_in_vocab
        return first_name_is_in_vocab and last_name_is_in_vocab


def load_mimic_iii_dummy_phi(
    mode: str, code: str, logger: logging.Logger
) -> pd.DataFrame:
    """
    mode: 'hospital' or 'shadow'
    code: "c1p2", "c1p1", "c1p0", "c0p2", "c0p1", "c0p0"
    """
    corpus_dir = get_repo_dir() / "corpus"
    path = corpus_dir / f"MIMIC_III_DUMMY_PHI_{mode.upper()}_{code.upper()}.csv"

    logger.info(f"Loading {path} ...")

    try:
        df = pd.read_csv(path, quoting=0)
        logger.info("Loaded!")
        return df
    except FileNotFoundError:
        logger.error(
            f"File does not exist ({path}): Please run split_mimic_iii_dummy_phi.py first"
        )


def get_save_path(mode: str, code: str) -> pathlib.PosixPath:
    """
    mode:Literal['hospital','shadow']
    code:Literal['c1p2','c1p1','c1p0','c0p2','c0p1','c0p0']
    """
    save_dir = get_repo_dir() / "corpus"
    save_basename = f"full_name_mentions_{mode.lower()}_{code.lower()}.tsv"
    save_path = save_dir / save_basename
    return save_path


if __name__ == "__main__":
    main()

import logging
import pathlib

import pandas as pd

from kart.src.modules.logging.logger import get_stream_handler
from kart.src.modules.privacy.utils.full_name_mentions import add_full_name_columns
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
        for code in ("c1p2", "c1p1", "c1p0", "c0p2", "c0p1", "c0p0"):
            df = load_mimic_iii_dummy_phi(mode, code, logger=logger)
            logger.info(f"Loaded DataFrame (n={len(df)})")

            df = add_full_name_columns(df, mode)

            df = df.query("patient_full_name_tfreq > 0")
            logger.info(f"Extracted records with full name mentions (n={len(df)})")

            df = df.drop_duplicates(subset="full_name_mention")
            logger.info(
                f"Extracted records with unique full name mentions (n={len(df)})"
            )

            judger = InVocabJudger()
            df = df.loc[
                df["patient_full_name"].apply(
                    lambda full_name: judger.first_and_last_names_are_in_vocab(
                        eval(str(full_name))[0], eval(str(full_name))[1]
                    )
                ),
                :,
            ]
            logger.info(
                f"Extracted records in which first & last names are expressed as one token (n={len(df)})"
            )

            df = df.drop_duplicates(subset="SUBJECT_ID")
            logger.info(f"Picked up one record per one subject (n={len(df)})")

            save_tsv(df, mode=mode, code=code, logger=logger)


class InVocabJudger:
    def __init__(self):
        self.namebook = PopularNameBook()

    def first_and_last_names_are_in_vocab(
        self, first_name: str, last_name: str
    ) -> bool:
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


def save_tsv(df: pd.DataFrame, mode: str, code: str, logger: logging.Logger):
    """
    mode:Literal['hospital','shadow']
    code:Literal['c1p2','c1p1','c1p0','c0p2','c0p1','c0p0']
    """
    save_dir = get_repo_dir() / "corpus"
    save_basename = f"full_name_mentions_{mode.lower()}_{code.lower()}.tsv"
    save_path = save_dir / save_basename
    df.to_csv(save_path, index=False, sep="\t")
    logger.info(f"Saved to {save_path}!")


if __name__ == "__main__":
    main()

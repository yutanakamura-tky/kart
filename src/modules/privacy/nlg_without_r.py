import argparse
import logging
import pathlib
import re
from typing import Optional

import pandas as pd
import torch
from transformers import BertConfig, BertForPreTraining, BertTokenizer

from kart.src.modules.logging.logger import get_file_handler, get_stream_handler
from kart.src.modules.privacy.utils.full_name_mentions import regexp_for_name
from kart.src.modules.privacy.utils.generator import Generator
from kart.src.modules.privacy.utils.path import get_repo_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = get_stream_handler()
file_handler = get_file_handler(f"{pathlib.Path(__file__).stem}.log")
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def main():
    args = get_args()
    logger.info(vars(args))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    logger.info("Loading BERT model ...")
    model = load_bert_model(args.model_code, hipaa=args.hipaa)
    model.eval()

    sample_size_config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }

    generation_config = {
        "top_k": args.top_k,
        "temperature": args.temperature,
        "burnin": args.burn_in,
        "max_iter": args.max_iter,
        "sample": True,
        "leed_out_len": 5,
        "generation_mode": "parallel-sequential",
    }

    print_config = {
        "print_every_iter": args.print_every_iter,
        "print_every_batch": 1,
        "verbose": not args.quiet,
    }

    logger.info(sample_size_config)
    logger.info(generation_config)
    logger.info(print_config)

    out_path = get_save_path(args)

    if args.use_full_name_knowledge or args.use_corpus:
        patient_info_path = (
            get_repo_dir() / f"corpus/gold_full_names_hospital_{args.model_code}.tsv"
        )

        df_patient_info = pd.read_csv(patient_info_path, sep="\t")
        full_names = df_patient_info["patient_full_name"].values.tolist()
        full_name_mentions = df_patient_info["full_name_mention"].values.tolist()

        if args.use_full_name_knowledge:
            prompts = [get_prompt(full_name=full_name) for full_name in full_names]

        else:
            prompts = [
                get_prompt(full_name_mention=full_name_mention)
                for full_name_mention in full_name_mentions
            ]

        with torch.cuda.device(args.cuda_device_number):
            model.to("cuda")
            Generator.generate(
                model=model,
                tokenizer=tokenizer,
                seed_texts=prompts,
                **sample_size_config,
                **generation_config,
                **print_config,
                use_cuda=True,
                logger=logger,
                out_path=out_path,
            )

    else:
        prompt = get_prompt(full_name=None)

        with torch.cuda.device(args.cuda_device_number):
            model.to("cuda")
            Generator.generate(
                model=model,
                tokenizer=tokenizer,
                seed_texts=prompt,
                **sample_size_config,
                **generation_config,
                **print_config,
                use_cuda=True,
                logger=logger,
                out_path=out_path,
            )

    logger.info(f"Sentences saved to {out_path}")


def get_save_path(args: argparse.Namespace) -> pathlib.PosixPath:
    out_dir = get_repo_dir() / "src/modules/privacy"
    out_basename = (
        "generation_result_"
        + f"model{'_'+args.model_code if args.model_code else ''}_"
        + f"iter_{args.max_iter}_"
        + f"batchsize_{args.batch_size}_"
        + f"temp_{args.temperature}_topk_{args.top_k}_burnin_{args.burn_in}_len_{args.max_length}"
    )

    if args.use_full_name_knowledge:
        out_basename += "_fullname_known"
    else:
        out_basename += "_fullname_unknown"

    if args.use_corpus:
        out_basename += "_corpus_used"
    else:
        out_basename += "_corpus_unused"

    if args.hipaa:
        out_basename += "_hipaa.txt"
    else:
        out_basename += "_no_anonymization.txt"

    out_path = out_dir / out_basename
    return out_path


def load_bert_model(
    model_code: Optional[str], hipaa: bool = False
) -> BertForPreTraining:
    if model_code is None:
        model = BertForPreTraining.from_pretrained("bert-base-uncased")
    else:
        model_dir = get_repo_dir() / (
            f"models/tf_bert_scratch_hospital_{model_code}_"
            + f"{'hipaa' if hipaa else 'no_anonymization'}/pretraining_output_stage1"
        )
        model_path = model_dir / "model.ckpt-1000000"
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertForPreTraining(config)
        model.load_tf_weights(config, model_path)
    return model


def get_prompt(full_name: Optional[str], full_name_mention: Optional[str]) -> str:

    if full_name_mention:
        prompt = convert_full_name_mention_to_prompt(full_name_mention)

    else:
        prompt = ""
        if full_name is None:
            prompt += f"{mask(2)}"
        else:
            prompt += full_name
        prompt += f" is a {mask(1)} year-old {mask(1)} presented with"
    return prompt


def convert_full_name_mention_to_prompt(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub(
        r"^" + regexp_for_name("first", False) + " " + regexp_for_name("last", False),
        "[MASK] [MASK]",
        text,
    )
    text = re.sub(r"\[\*\*.+?\*\*\]", "", text)
    return text


def mask(length: int) -> str:
    return " ".join(["[MASK]"] * length)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--use-full-name-knowledge",
        dest="use_full_name_knowledge",
        action="store_true",
        help="If set True, the prior knowledge (full names of the subjects) will be used for privacy attack.",
    )
    parser.add_argument(
        "--use-corpus",
        dest="use_corpus",
        action="store_true",
        help="If set True, the anonymized version of the pre-training data will be used for privacy attack.",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--model-code",
        dest="model_code",
        type=str,
        nargs="?",
        help="Specify BERT model code ('c1p2', 'c1p1', 'c1p0', 'c0p2', 'c0p1', 'c0p0')",
    )
    parser.add_argument(
        "--hipaa",
        dest="hipaa",
        action="store_true",
        help="If set True, BERT model pre-trained with anonymized data will be used",
    )
    parser.add_argument("-n", "--n-samples", dest="n_samples", type=int, default=100)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("-l", "--max-length", dest="max_length", type=int, default=256)
    parser.add_argument(
        "-C", "--cuda-device-number", dest="cuda_device_number", type=int, default=1
    )
    parser.add_argument("-k", "--top-k", dest="top_k", type=int, default=100)
    parser.add_argument(
        "-t", "--temperature", dest="temperature", type=float, default=1.0
    )
    parser.add_argument("-r", "--burn-in", dest="burn_in", type=int, default=250)
    parser.add_argument("-i", "--max-iter", dest="max_iter", type=int, default=500)
    parser.add_argument(
        "-p", "--print-every-iter", dest="print_every_iter", type=int, default=50
    )
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

import argparse
import logging
import pathlib
from typing import Optional

import pandas as pd
import torch
from transformers import BertConfig, BertForPreTraining, BertTokenizer

from kart.src.modules.logging.logger import get_file_handler, get_stream_handler
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

    if args.full_name_source:
        full_names = pd.read_csv(args.full_name_source, sep="\t")[
            "patient_full_name"
        ].values.tolist()
        prompts = [
            get_prompt(
                full_name=full_name, chief_complaint_length=args.chief_complaint_length
            )
            for full_name in full_names
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
        prompt = get_prompt(
            full_name=None, chief_complaint_length=args.chief_complaint_length
        )

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

    if args.full_name_source:
        out_basename += "_fullname_known"
    else:
        out_basename += "_fullname_unknown"

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
        model_dir = (
            get_repo_dir() / f"models/tf_bert_scratch_hospital_{model_code}_"
            + f"{'hipaa' if hipaa else 'no_anonymization'}/pretraining_output_stage1"
        )
        model_path = model_dir / "model.ckpt-1000000"
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertForPreTraining(config)
        model.load_tf_weights(config, model_path)
    return model


def get_prompt(full_name: Optional[str], chief_complaint_length: int) -> str:
    prompt = ""

    if full_name is None:
        prompt += f"{mask(2)}"
    else:
        prompt += full_name

    prompt += f" is a {mask(1)} year-old {mask(1)} presented with {mask(chief_complaint_length)}. "
    prompt += "The patient has a history of"
    return prompt


def mask(length: int) -> str:
    return " ".join(["[MASK]"] * length)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--full-name-source",
        dest="full_name_source",
        type=str,
        nargs="?",
        help="To simulate that the attacker already knows full names of the subjects, "
        + "specify path of TSV files containing patient full names in 'patient_full_name' column."
        + "To simulate that the attacker does not know full names of the subjects, leave this blank.",
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
    parser.add_argument(
        "-c",
        "--chief-complaint-length",
        dest="chief_complaint_length",
        type=int,
        default=110,
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

import argparse
import logging
from typing import Optional

import torch
from transformers import BertForPreTraining, BertTokenizer

from kart.src.modules.logging.logger import get_stream_handler
from kart.src.modules.privacy.utils.generator import Generator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = get_stream_handler()
file_handler = logging.FileHandler("nlg_with_bert.log")
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def main():
    args = get_args()
    logger.info(vars(args))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForPreTraining.from_pretrained("bert-base-uncased")
    model.eval()

    seed_text = get_prompt(
        full_name=None, chief_complaint_length=args.chief_complaint_length
    )
    logger.info(f"seed text: {seed_text}")

    sample_size_config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "max_len": args.max_length,
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

    print_config = {"print_every_iter": 50, "print_every_batch": 1, "verbose": True}

    with torch.cuda.device(args.cuda_device_number):
        model.to("cuda")
        bert_sents = Generator.generate(
            model=model,
            tokenizer=tokenizer,
            seed_text=seed_text,
            **sample_size_config,
            **generation_config,
            **print_config,
            cuda=True,
        )

    with open("generation_result.txt", "w") as f:
        f.writelines(bert_sents)


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


def get_args():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

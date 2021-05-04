import argparse
import logging
from typing import Optional

import torch
from transformers import BertConfig, BertForPreTraining, BertTokenizer

from kart.src.modules.logging.logger import get_stream_handler
from kart.src.modules.privacy.utils.generator import Generator
from kart.src.modules.privacy.utils.path import get_repo_dir

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

    logger.info("Loading BERT model ...")
    model = load_bert_model(args.model_code)
    model.eval()

    seed_text = get_prompt(
        full_name=None, chief_complaint_length=args.chief_complaint_length
    )
    logger.info(f"seed text: {seed_text}")

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

    out_path = (
        "generation_result_"
        + f"model{'_'+args.model_code if args.model_code else ''}_"
        + f"iter_{args.max_iter}_"
        + f"batchsize_{args.batch_size}_"
        + f"temp_{args.temperature}_topk_{args.top_k}_burnin_{args.burn_in}_len_{args.max_length}.txt"
    )

    with torch.cuda.device(args.cuda_device_number):
        model.to("cuda")
        Generator.generate(
            model=model,
            tokenizer=tokenizer,
            seed_text=seed_text,
            **sample_size_config,
            **generation_config,
            **print_config,
            use_cuda=True,
            logger=logger,
            out_path=out_path,
        )

    logger.info(f"Sentences saved to {out_path}")


def load_bert_model(model_code: Optional[str]) -> BertForPreTraining:
    if model_code is None:
        model = BertForPreTraining.from_pretrained("bert-base-uncased")
    else:
        model_dir = (
            get_repo_dir()
            / f"models/tf_bert_hospital_{model_code}/pretraining_output_stage1"
        )
        model_path = model_dir / "model.ckpt-100000"
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


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model", "--model-code", dest="model_code", type=str, nargs="?"
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

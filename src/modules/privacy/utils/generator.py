import copy
import logging
import math
import pathlib
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers


class SeedTextProcessor:
    @classmethod
    def seed_text_to_token_ids(
        cls,
        seed_text: str,
        tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer,
        max_length: int,
        batch_size: int,
    ) -> Tuple[List[List[int]], int]:
        batch: List[str] = cls.duplicate_seed_text(seed_text, batch_size)
        initial_tokenization_result: Dict = tokenizer.batch_encode_plus(
            batch, padding="max_length", max_length=max_length
        )
        initial_input_ids = initial_tokenization_result["input_ids"]

        # Note: seed_length does not include '[CLS]' and '[SEP]' tokens
        seed_length = sum(initial_tokenization_result["attention_mask"][0]) - 2
        if seed_length == max_length - 2:
            return (initial_input_ids, seed_length)
        else:
            padded_batch: List[str] = cls.pad_with_mask_tokens(
                batch, max_length - seed_length - 2
            )
            padded_tokenization_result: Dict = tokenizer.batch_encode_plus(
                padded_batch, padding="max_length", max_length=max_length
            )
            padded_input_ids = padded_tokenization_result["input_ids"]
            return (padded_input_ids, seed_length)

    @staticmethod
    def duplicate_seed_text(seed_text: str, batch_size: int) -> List[str]:
        batch: List[str] = [seed_text for _ in range(batch_size)]
        return batch

    @staticmethod
    def pad_with_mask_tokens(batch: List[str], padding_length: int) -> List[str]:
        result = [seed_text + " [MASK]" * padding_length for seed_text in batch]
        return result


class Generator:
    @classmethod
    def generate(
        cls,
        n_samples: int,
        model: transformers.models.bert.modeling_bert.BertForPreTraining,
        tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer,
        seed_texts: Union[str, List[str]],
        out_path: Union[str, pathlib.PosixPath],
        batch_size: int = 10,
        max_length: int = 25,
        leed_out_len: int = 15,
        generation_mode: str = "parallel-sequential",
        sample: bool = True,
        top_k: int = 100,
        temperature: Optional[float] = 1.0,
        temperature_scheduler: Optional[Callable[[int], float]] = None,
        burnin: int = 200,
        max_iter: int = 500,
        use_cuda: bool = False,
        print_every_batch: int = 1,
        print_every_iter: int = 50,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> List[str]:
        sentences = []
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()

        for batch_n in range(n_batches):
            if type(seed_texts) is str:
                seed_text = seed_texts
            elif type(seed_texts) is list:
                seed_ix = batch_n % len(seed_texts)
                seed_text = seed_texts[seed_ix]

            if logger:
                logger.info(f"Seed text: {seed_text}")

            if generation_mode == "parallel-sequential":
                batch = cls.parallel_sequential_generation(
                    seed_text,
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    max_length=max_length,
                    top_k=top_k,
                    temperature=temperature,
                    temperature_scheduler=temperature_scheduler,
                    burnin=burnin,
                    max_iter=max_iter,
                    use_cuda=use_cuda,
                    verbose=verbose,
                    print_every_iter=print_every_iter,
                    sample=sample,
                )

            if (batch_n + 1) % print_every_batch == 0:
                print(
                    "Finished batch %d in %.3fs"
                    % (batch_n + 1, time.time() - start_time)
                )
                start_time = time.time()

            with open(out_path, "a") as f:
                f.write("\n".join(batch) + "\n")

            sentences += batch
        return sentences

    @classmethod
    def parallel_sequential_generation(
        cls,
        seed_text: str,
        model: transformers.models.bert.modeling_bert.BertForPreTraining,
        tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer,
        batch_size: int,
        max_length: int,
        top_k: int,
        temperature: float,
        temperature_scheduler: Optional[Callable[[int], float]],
        max_iter: int,
        burnin: int,
        use_cuda: bool,
        print_every_iter: int,
        verbose: bool,
        sample: bool,
        logger: logging.Logger,
    ) -> List[str]:
        """Generate for one random position at a timestep

        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        MASK_ID = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        original_input_ids, seed_length = SeedTextProcessor.seed_text_to_token_ids(
            seed_text, tokenizer, max_length, batch_size
        )
        input_ids = copy.deepcopy(original_input_ids)

        for ii in range(max_iter):
            while True:
                target_position = np.random.randint(1, max_length - 1)
                if original_input_ids[0][target_position] == MASK_ID:
                    break
                else:
                    continue
            for jj in range(batch_size):
                input_ids[jj][target_position] = MASK_ID
            inp = (
                torch.tensor(input_ids).cuda() if use_cuda else torch.tensor(input_ids)
            )
            logits = model(inp)[0]
            topk = top_k if (ii >= burnin) else 0
            if temperature_scheduler is None:
                temperature = temperature
            else:
                temperature = temperature_scheduler(ii)
            new_token_ids = cls.generate_step(
                out=logits,
                target_position=target_position,
                top_k=topk,
                temperature=temperature,
                sample=(ii < burnin),
            )
            for jj in range(batch_size):
                input_ids[jj][target_position] = new_token_ids[jj]

            if verbose and np.mod(ii + 1, print_every_iter) == 0:
                logger.info(f'{"="*30} Iter {ii+1} {"="*30}')
                if logger:
                    logger.info(
                        f"Number of [MASK] tokens: {sum(map(lambda x: x==MASK_ID, input_ids[0]))}"
                    )
                logger.info([tokenizer.decode(token_ids) for token_ids in input_ids])

        return [tokenizer.decode(token_ids) for token_ids in input_ids]

    @staticmethod
    def generate_step(
        out: torch.Tensor,
        target_position: int,
        temperature: Optional[float],
        top_k: int,
        sample: bool,
        return_list: bool = True,
    ):
        """Generate a word from from out[target_position]

        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - target_position (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k
        """
        logits = out[:, target_position]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx

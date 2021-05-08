import logging
import math
import pathlib
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import BertForPreTraining, BertTokenizer


class SeedTextProcessor:
    @classmethod
    def seed_text_to_token_ids(
        cls,
        seed_text: str,
        tokenizer: BertTokenizer,
        max_length: int,
        batch_size: int,
    ) -> Tuple[List[List[int]], int]:
        batch: List[str] = cls.duplicate_seed_text(seed_text, batch_size)
        initial_tokenization_result: Dict = tokenizer.batch_encode_plus(
            batch, padding="max_length", max_length=max_length
        )
        initial_input_ids = initial_tokenization_result["input_ids"]

        # Note: seed_length does not include '[CLS]' and '[SEP]' tokens
        initial_seed_length = sum(initial_tokenization_result["attention_mask"][0]) - 2
        if initial_seed_length == max_length - 2:
            return (initial_input_ids, initial_seed_length)
        elif initial_seed_length > max_length - 2:
            # When seed text is too long, truncate
            truncated_input_ids = [
                input_ids[: max_length - 1] + [input_ids[-1]]
                for input_ids in initial_input_ids
            ]
            seed_length = max_length - 2
            return (truncated_input_ids, seed_length)
        elif initial_seed_length < max_length - 2:
            # When seed text is short, pad with [MASK] tokens
            padded_batch: List[str] = cls.pad_with_mask_tokens(
                batch, max_length - initial_seed_length - 2
            )
            padded_tokenization_result: Dict = tokenizer.batch_encode_plus(
                padded_batch, padding="max_length", max_length=max_length
            )
            padded_input_ids = padded_tokenization_result["input_ids"]
            return (padded_input_ids, initial_seed_length)

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
        model: BertForPreTraining,
        tokenizer: BertTokenizer,
        seed_texts: Union[str, List[str]],
        out_path: Union[str, pathlib.PosixPath],
        batch_size: int = 10,
        max_length: int = 25,
        sequential: str = "never",
        leed_out_len: int = 15,
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
        """
        sequential: ('always', 'first', 'never')
        - 'always': Always generated in L->R order.
        - 'first': First generated in L->R order till the end. Generated in random positions afterwards.
        - 'never': Always generated in random positions.
        """

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

            batch = cls.parallel_sequential_generation(
                seed_text,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                sequential=sequential,
                top_k=top_k,
                temperature=temperature,
                temperature_scheduler=temperature_scheduler,
                burnin=burnin,
                max_iter=max_iter,
                use_cuda=use_cuda,
                verbose=verbose,
                print_every_iter=print_every_iter,
                sample=sample,
                logger=logger,
            )

            if (batch_n + 1) % print_every_batch == 0:
                logger.info(
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
        model: BertForPreTraining,
        tokenizer: BertTokenizer,
        batch_size: int,
        max_length: int,
        sequential: str,
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
        """
        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        MASK_ID = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        input_ids, seed_length = SeedTextProcessor.seed_text_to_token_ids(
            seed_text, tokenizer, max_length, batch_size
        )
        writable_positions = [
            ix for ix in range(len(input_ids[0])) if input_ids[0][ix] == MASK_ID
        ]

        for ii in range(max_iter):
            if sequential == "always" or (
                sequential == "first" and ii < len(writable_positions)
            ):
                target_position = writable_positions[ii % len(writable_positions)]
            else:
                target_position = writable_positions[
                    np.random.randint(0, len(writable_positions))
                ]
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
                    n_mask_tokens = (np.array(input_ids) == MASK_ID).sum(axis=1)
                    logger.info(
                        "Average number of [MASK] tokens: " + f"{n_mask_tokens.mean()}"
                    )
                logger.info(tokenizer.decode(input_ids[0]))

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

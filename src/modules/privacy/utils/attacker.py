import gc
import re

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm.notebook import tqdm

from .full_name_mentions import regexp_for_name


class MlmAttacker:
    """
    Examples
    --------
    >>> from transformers import BertTokenizer, BertForPreTraining
    >>> from . import DummyPhiMap, PopularNameBook
    >>>
    >>> namebook = PopularNameBook()
    >>> candidate_full_names = []
    >>> for first_name in namebook.first_names_in_vocab:
    >>>     for last_name in namebook.last_names_in_vocab:
    >>>         candidate_full_names.append((first_name, last_name))
    >>>
    >>> phimap = DummyPhiMap()
    >>> bert_model = BertForPreTraining.from_pretrained('bert-base-uncased')
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>>
    >>> attacker = MlmAttacker(bert_model, tokenizer, namebook, phimap)
    >>> texts = ['[**Known firstname 10079**] [**Last Name (LF) 18539**] is a 42 year-old female']
    >>> result = attacker.from_texts_and_names(texts, candidate_full_names)
    """

    def __init__(self, bert_model, tokenizer, namebook, surmap, mode="hospital"):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.namebook = namebook  # namebook: PopularNameBook object
        self.surmap = surmap  # surmap: DummyPhiMap object
        self.mode = mode

    @staticmethod
    def replace_name_placeholders(target, to, text):
        """
        target: 'first' or 'last'
        """
        return re.sub(regexp_for_name(target, False), to, text)

    def full_name_placeholders_to_bert_mask_for_single_text(
        self, text, ignore_subwords=True
    ):
        if ignore_subwords:
            text = re.sub(regexp_for_name("first", False), "[MASK]", text)
            text = re.sub(regexp_for_name("last", False), "[MASK]", text)
        else:
            for first_name_placeholder in re.findall(
                regexp_for_name("first", False), text
            ):
                first_name = getattr(self.surmap, self.mode)[first_name_placeholder]
                text = text.replace(
                    first_name_placeholder,
                    " ".join(["[MASK]"] * self.namebook.token_len[first_name]),
                )
            for last_name_placeholder in re.findall(
                regexp_for_name("last", False), text
            ):
                last_name = getattr(self.surmap, self.mode)[last_name_placeholder]
                text = text.replace(
                    last_name_placeholder,
                    " ".join(["[MASK]"] * self.namebook.token_len[last_name]),
                )
        return text

    def full_name_placeholders_to_bert_mask(self, texts, ignore_subwords=True):
        if type(texts) is list:
            return list(
                map(
                    lambda x: self.full_name_placeholders_to_bert_mask_for_single_text(
                        x, ignore_subwords
                    ),
                    texts,
                )
            )
        elif type(texts) is str:
            return self.full_name_placeholders_to_bert_mask_for_single_text(
                texts, ignore_subwords
            )

    def to_bert_input(self, texts):
        if type(texts) is list:
            bert_input = texts
        elif type(texts) is str:
            bert_input = [texts]

        encoded = self.tokenizer.batch_encode_plus(
            bert_input, max_length=self.tokenizer.max_len, pad_to_max_length=True
        )
        return {
            k: torch.tensor(v).to(self.bert_model.device) for k, v in encoded.items()
        }

    def get_attack_result(self, mlm_output, normalize_with_names_in_vocab=True):
        """
        Parameters
        ----------
        mlm_output: torch.Tensor
            3rd order tensor (n_batch, length, n_vocab)
        normalize_with_names_in_vocab: bool
        """
        # 2nd order tensor (n_batch, n_vocab)
        first_token_output = mlm_output[:, 1, :].to(self.bert_model.device)
        second_token_output = mlm_output[:, 2, :].to(self.bert_model.device)

        first_token_prob = F.softmax(first_token_output, dim=1)
        second_token_prob = F.softmax(second_token_output, dim=1)

        if normalize_with_names_in_vocab:
            first_name_vocab_ix = list(self.namebook.first_name_vocab_wtoi.values())
            last_name_vocab_ix = list(self.namebook.last_name_vocab_wtoi.values())

            # 2nd order tensor (n_batch, n_first_names_in_vocab)
            first_name_prob = first_token_prob[
                :, torch.tensor(first_name_vocab_ix).to(self.bert_model.device)
            ]

            # 2nd order tensor (n_batch, n_last_names_in_vocab)
            last_name_prob = second_token_prob[
                :, torch.tensor(last_name_vocab_ix).to(self.bert_model.device)
            ]

            first_name_prob /= first_name_prob.sum()
            last_name_prob /= last_name_prob.sum()
            return (first_name_prob.cpu(), last_name_prob.cpu())

        else:
            return (first_token_prob.cpu(), second_token_prob.cpu())

    def from_texts(
        self, texts, batch_size=16, random_state=42, normalize_with_names_in_vocab=True
    ):
        """
        Parameters
        ----------
        texts: list[str] or str
        random_state: int
        normalize_with_names_in_vocab: bool
        """
        masked_texts = self.full_name_placeholders_to_bert_mask(texts)
        batches = Mlm.to_bert_input(
            self.tokenizer, masked_texts, batch_size=batch_size, pad_to_max_length=True
        )

        first_name_prob = None
        last_name_prob = None

        for batch in tqdm(batches):
            mlm_output = Mlm.mlm_on_batch(
                self.bert_model,
                batch,
                random_state=random_state,
                softmax=False,
            )
            batch_first_name_prob, batch_last_name_prob = self.get_attack_result(
                mlm_output, normalize_with_names_in_vocab
            )
            if first_name_prob is None:
                first_name_prob = batch_first_name_prob
            else:
                first_name_prob = torch.cat([first_name_prob, batch_first_name_prob])

            if last_name_prob is None:
                last_name_prob = batch_last_name_prob
            else:
                last_name_prob = torch.cat([last_name_prob, batch_last_name_prob])

        return (first_name_prob, last_name_prob)

    def from_texts_and_names(
        self, texts, full_names, random_state=42, normalize_with_names_in_vocab=True
    ):
        """
        Parameters
        ----------
        texts: list[str] or str
        full_names: list[tuple]
            target names in tuple [(first_name_1, last_name_1), ...]
        random_state: int
        normalize_with_names_in_vocab: bool
        """
        first_name_prob, last_name_prob = self.from_texts(
            texts, random_state, normalize_with_names_in_vocab
        )

        first_name_incides = np.array(
            [
                self.namebook.first_names_in_vocab.index(full_name[0])
                for full_name in full_names
            ]
        )
        last_name_incides = np.array(
            [
                self.namebook.last_names_in_vocab.index(full_name[1])
                for full_name in full_names
            ]
        )

        first_name_probs = first_name_prob[:, first_name_incides]
        last_name_probs = last_name_prob[:, last_name_incides]

        name_probs = first_name_probs * last_name_probs

        return name_probs


class PerplexityCalculator:
    """
    Examples
    --------
    >>> from transformers import BertTokenizer, BertForPreTraining
    >>> bert_model = BertForPreTraining.from_pretrained('bert-base-uncased')
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> ppc = PerplexityCalculator(bert_model, tokenizer)
    >>> ppc.ppl('there is a book on the desk')
    3.9378058910369873
    """

    def __init__(self, bert_model, tokenizer):
        self.bert_model = bert_model
        self.tokenizer = tokenizer

    @staticmethod
    def get_stride_masked_sentences(tokens_of_single_sentence):
        results = [
            [token for token in tokens_of_single_sentence]
            for _ in range(len(tokens_of_single_sentence))
        ]

        for i in range(len(tokens_of_single_sentence)):
            results[i][i] = "[MASK]"

        return results

    def tokens_to_bert_input(
        self, tokens_of_sentences, batch_size, pad_to_max_length=False
    ):
        if type(tokens_of_sentences[0]) is str:
            tokens_of_sentences = [tokens_of_sentences]
            n_samples = 1
        elif type(tokens_of_sentences[0]) is list:
            n_samples = len(tokens_of_sentences)

        encoded = self.tokenizer.batch_encode_plus(
            tokens_of_sentences,
            pad_to_max_length=pad_to_max_length,
            max_length=self.tokenizer.max_len,
        )
        whole_bert_input = {k: torch.tensor(v) for k, v in encoded.items()}
        returns = []
        j = 0
        while batch_size * j < n_samples:
            input_to_append = {
                k: v[batch_size * j : batch_size * (j + 1)]
                for k, v in whole_bert_input.items()
            }
            returns.append(input_to_append)
            j += 1
        return returns

    def mlm(self, batches, softmax=True, random_state=42):
        """
        Parameters
        ----------
        batches: list[dict]
            List of dicts obtained by transformers.BertTokenizer.batch_encode_plus()
        """
        result_list = []
        with torch.no_grad():
            for batch in tqdm(batches):
                seed_everything(random_state)
                bert_input = {k: v.to(self.bert_model.device) for k, v in batch.items()}
                mlm_logits = self.bert_model(**bert_input)[0]
                if softmax:
                    mlm_probs = F.softmax(mlm_logits, dim=2).to("cpu")
                    del mlm_logits
                    gc.collect()
                    result_list.append(mlm_probs)
                else:
                    mlm_logits = mlm_logits.to("cpu")
                    result_list.append(mlm_logits)
        return torch.cat(result_list, dim=0)

    def ppl(self, text, batch_size=8):
        # list (n_len)
        tokens = self.tokenizer.tokenize(text)
        # list of list (n_len, n_len)
        stride_masked_sentences = self.get_stride_masked_sentences(tokens)
        # list of dict (ceil(n_len / batch_size))
        bert_inputs = self.tokens_to_bert_input(stride_masked_sentences, batch_size)
        # torch.Tensor (n_len, max_len, hidden_dim)
        mlm_probs = self.mlm(bert_inputs, softmax=True)

        # torch.Tensor (n_len, n_len, hidden_dim)
        # extract between [CLS] and [PAD], then exclude [CLS] and [PAD]
        n_sample = mlm_probs.size()[0]
        mlm_probs = mlm_probs[:, 1 : n_sample + 1, :]

        # torch.Tensor (n_len, hidden_dim)
        mlm_probs = torch.diagonal(mlm_probs).transpose(0, 1)
        mlm_probs = mlm_probs.to(self.bert_model.device)
        token_ids_of_original_text = torch.tensor(
            self.tokenizer.encode(text, max_length=self.tokenizer.max_len)[1:-1]
        ).to(self.bert_model.device)
        # torch.Tensor (n_len)
        token_probs = torch.gather(
            mlm_probs, dim=1, index=token_ids_of_original_text.unsqueeze(-1)
        ).squeeze(1)
        log_ppl = token_probs.log().sum() * (-1) / len(tokens)
        ppl = log_ppl.exp().item()

        del mlm_probs, token_probs, token_ids_of_original_text, log_ppl
        gc.collect()

        return ppl


class Mlm:
    """
    Examples
    --------
    >>> from transformers import BertTokenizer, BertForPreTraining
    >>> bert_model = BertForPreTraining.from_pretrained('bert-base-uncased')
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> text = 'The quick brown fox jumps over the lazy dog'
    >>> mlm_output = Mlm.mlm(bert_model, tokenizer, text)
    """

    def __init__(self):
        pass

    @classmethod
    def mlm(
        cls,
        bert_model,
        tokenizer,
        sentences,
        batch_size=8,
        random_state=42,
        pad_to_max_length=True,
        softmax=True,
    ):
        """
        Parameters
        ----------
        bert_model: transformers.BertForPreTraining
        tokenizer: transformers.BertTokenizer
        sentences: str or list[str]
        batch_size: int
        random_state: int
        pad_to_max_length: bool
        softmax: bool
        """
        batches = cls.to_bert_input(tokenizer, sentences, batch_size, pad_to_max_length)
        result_list = []
        for batch in tqdm(batches):
            result_list.append(
                cls.mlm_on_batch(bert_model, batch, random_state, softmax)
            )
        return torch.cat(result_list, dim=0)

    @staticmethod
    def mlm_on_batch(bert_model, batch, random_state, softmax):
        """
        Parameters
        ----------
        bert_model: transformers.BertForPreTraining
        batches: dict
            Dictionary obtained by transformers.BertTokenizer.batch_encode_plus()
        random_state: int
        softmax: bool
        """
        with torch.no_grad():
            seed_everything(random_state)
            bert_input = {k: v.to(bert_model.device) for k, v in batch.items()}
            mlm_logits = bert_model(**bert_input)[0]
            if softmax:
                mlm_probs = F.softmax(mlm_logits, dim=2).to("cpu")
                del mlm_logits
                gc.collect()
                return mlm_probs
            else:
                mlm_logits = mlm_logits.to("cpu")
                return mlm_logits

    @classmethod
    def to_bert_input(cls, tokenizer, sentences, batch_size, pad_to_max_length=True):
        if type(sentences) is str:
            sentences = [sentences]

        encoded = tokenizer.batch_encode_plus(
            sentences,
            pad_to_max_length=pad_to_max_length,
            max_length=tokenizer.max_len,
        )
        whole_bert_input = {k: torch.tensor(v) for k, v in encoded.items()}
        return cls.chunk_bert_input(whole_bert_input, batch_size)

    @staticmethod
    def chunk_bert_input(whole_bert_input, batch_size):
        n_samples = whole_bert_input["input_ids"].size()[0]
        batches = []
        j = 0
        while batch_size * j < n_samples:
            input_to_append = {
                k: v[batch_size * j : batch_size * (j + 1)]
                for k, v in whole_bert_input.items()
            }
            batches.append(input_to_append)
            j += 1
        return batches

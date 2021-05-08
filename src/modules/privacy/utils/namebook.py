import csv
from collections import OrderedDict

from faker.providers.person.en_US import Provider
from transformers import BertTokenizer


class PopularNameBook:
    def __init__(self, tokenizer_code="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_code)

        self.first_name_popularity_female = Provider.first_names_female
        self.first_name_popularity_male = Provider.first_names_male
        self.first_name_popularity = self.merge_female_and_male_popularity_dicts(
            self.first_name_popularity_female, self.first_name_popularity_male
        )
        self.first_names = list(self.first_name_popularity.keys())
        self.first_name_token_len = {
            first_name: len(self.tokenizer.tokenize(first_name))
            for first_name in self.first_names
        }

        self.first_names_in_vocab = self.pick_up_words_in_vocab(
            self.first_names, self.tokenizer
        )

        self.first_name_vocab_wtoi = self.create_vocab_map(
            self.first_names, self.tokenizer
        )
        self.first_name_wtoi = {name: i for i, name in enumerate(self.first_names)}

        _first_name_popularity_in_vocab_female = OrderedDict(
            [
                (k, v)
                for k, v in self.first_name_popularity_female.items()
                if k in self.first_names_in_vocab
            ]
        )
        _first_name_popularity_in_vocab_female_sum = sum(
            list(_first_name_popularity_in_vocab_female.values())
        )
        self.first_name_popularity_in_vocab_female = OrderedDict(
            [
                (k, v / _first_name_popularity_in_vocab_female_sum)
                for k, v in _first_name_popularity_in_vocab_female.items()
            ]
        )

        _first_name_popularity_in_vocab_male = OrderedDict(
            [
                (k, v)
                for k, v in self.first_name_popularity_male.items()
                if k in self.first_names_in_vocab
            ]
        )
        _first_name_popularity_in_vocab_male_sum = sum(
            list(_first_name_popularity_in_vocab_male.values())
        )
        self.first_name_popularity_in_vocab_male = OrderedDict(
            [
                (k, v / _first_name_popularity_in_vocab_male_sum)
                for k, v in _first_name_popularity_in_vocab_male.items()
            ]
        )

        self.first_name_popularity_in_vocab = (
            self.merge_female_and_male_popularity_dicts(
                self.first_name_popularity_in_vocab_female,
                self.first_name_popularity_in_vocab_male,
            )
        )

        # Sort last names in alphabetical order
        self.last_name_popularity = OrderedDict(
            sorted(Provider.last_names.items(), key=lambda x: x[0])
        )
        self.last_names = list(self.last_name_popularity.keys())
        self.last_name_token_len = {
            last_name: len(self.tokenizer.tokenize(last_name))
            for last_name in self.last_names
        }
        self.last_names_in_vocab = self.pick_up_words_in_vocab(
            self.last_names, self.tokenizer
        )

        _last_name_popularity_in_vocab = OrderedDict(
            [
                (k, v)
                for k, v in self.last_name_popularity.items()
                if k in self.last_names_in_vocab
            ]
        )
        _last_name_popularity_in_vocab_sum = sum(
            list(_last_name_popularity_in_vocab.values())
        )
        self.last_name_popularity_in_vocab = OrderedDict(
            [
                (k, v / _last_name_popularity_in_vocab_sum)
                for k, v in _last_name_popularity_in_vocab.items()
            ]
        )

        self.last_name_vocab_wtoi = self.create_vocab_map(
            self.last_names, self.tokenizer
        )
        self.last_name_wtoi = {name: i for i, name in enumerate(self.last_names)}

        self.token_len = {}
        for name in self.first_names:
            self.token_len[name] = self.first_name_token_len[name]
        for name in self.last_names:
            self.token_len[name] = self.last_name_token_len[name]

        self.n_first_names = len(self.first_names)
        self.n_last_names = len(self.last_names)
        self.n_full_names = self.n_first_names * self.n_last_names

        self.n_first_names_in_vocab = len(self.first_names_in_vocab)
        self.n_last_names_in_vocab = len(self.last_names_in_vocab)
        self.n_full_names_in_vocab = (
            self.n_first_names_in_vocab * self.n_last_names_in_vocab
        )

    @staticmethod
    def merge_female_and_male_popularity_dicts(female_dict, male_dict):
        unique_keys = sorted(set(female_dict.keys()) | set(male_dict.keys()))
        result = OrderedDict([(key, 0.0) for key in unique_keys])
        for k, v in female_dict.items():
            result[k] += v * 0.5 / sum(list(female_dict.values()))
        for k, v in male_dict.items():
            result[k] += v * 0.5 / sum(list(male_dict.values()))
        return result

    def popularity(self, full_name="", in_vocab=False):
        first_name, last_name = full_name.split()
        if first_name == "*" and last_name != "*":
            if in_vocab:
                return self.last_name_popularity_in_vocab[last_name]
            else:
                return self.last_name_popularity[last_name]
        elif first_name != "*" and last_name == "*":
            if in_vocab:
                return self.first_name_popularity_in_vocab[first_name]
            else:
                return self.first_name_popularity[first_name]
        elif first_name != "*" and last_name != "*":
            if in_vocab:
                return (
                    self.first_name_popularity_in_vocab[first_name]
                    * self.last_name_popularity_in_vocab[last_name]
                )
            else:
                return (
                    self.first_name_popularity[first_name]
                    * self.last_name_popularity[last_name]
                )
        elif first_name == "*" and last_name == "*":
            return 1.0

    @staticmethod
    def pick_up_words_in_vocab(words, tokenizer):
        result = list(filter(lambda x: len(tokenizer.encode(x)) - 2 == 1, words))
        return result

    @staticmethod
    def create_vocab_map(words, tokenizer):
        result = {word: tokenizer.encode(word)[1] for word in words}
        return result


class NameBook:
    def __init__(self, names, tokenizer_code="bert-base-uncased"):
        self.names = names
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_code)
        self.names_in_vocab = self.pick_up_words_in_vocab(self.names, self.tokenizer)
        self.vocab_wtoi = self.create_vocab_map(self.names_in_vocab, self.tokenizer)

    def __get__(self, ix):
        return self.names[ix]

    def __repr__(self):
        result = "NameBook([" + ",".join([f"'{name}'" for name in self.names]) + "])"
        return result

    @classmethod
    def from_csv(cls, csv_path, tokenizer_code="bert-base-uncased"):
        with open(csv_path) as f:
            reader = csv.reader(f)
            names = [row[0] for row in reader]
        return cls(names, tokenizer_code)

    @staticmethod
    def pick_up_words_in_vocab(words, tokenizer):
        result = list(filter(lambda x: len(tokenizer.encode(x)) - 2 == 1, words))
        return result

    @staticmethod
    def create_vocab_map(words, tokenizer):
        result = {word: tokenizer.encode(word)[1] for word in words}
        return result

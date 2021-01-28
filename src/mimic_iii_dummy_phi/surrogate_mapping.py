import os
import random

import pandas as pd
from faker.factory import Factory


def add_surrogate_mapping(
    input_path,
    output_path,
    method_name,
    *method_args,
    random_state=42,
    postprocess=None,
    overwrite=False,
    tsv=False
):
    """
    input_path: str
    output_path: str
    method_name: str
        Faker library method name you want call to create surrogate entity.
        e.g., 'random_int'
    method_args: n_args
    postprocess: callable or None (optional)
        Postprocess to apply surrogate entity.
        e.g., lambda x: x.replace('\n', '')
    """

    if os.path.getsize(input_path) == 0:
        pass

    else:
        Faker = Factory.create
        fake = Faker()

        method = getattr(fake, method_name)

        df = pd.read_csv(input_path, header=None)

        random.seed(random_state)
        faker_seeds = [random.randint(0, 99999999) for i in range(len(df))]

        surrogates = []

        for i in range(len(df)):
            fake.seed(faker_seeds[i])
            surrogates.append(method(*method_args))

        if postprocess is not None:
            surrogates = list(map(postprocess, surrogates))

        df["surrogates"] = surrogates
        df.to_csv(
            output_path,
            header=False,
            index=False,
            sep="\t" if tsv else ",",
            mode="w" if overwrite else "a",
        )

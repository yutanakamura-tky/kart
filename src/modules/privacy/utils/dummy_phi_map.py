import pandas as pd


class DummyPhiMap:
    def __init__(self):
        self.hospital = self.create_map("hospital")
        self.shadow = self.create_map("shadow")

    def create_map(self, corpus_name):
        df_surmap = pd.read_csv(
            f"../corpus/dummy_phi/{corpus_name}/surrogate_map.csv",
            quoting=0,
            header=None,
        ).fillna("")
        surmap = {k: v for k, v in zip(df_surmap[0].values, df_surmap[1].values)}
        return surmap

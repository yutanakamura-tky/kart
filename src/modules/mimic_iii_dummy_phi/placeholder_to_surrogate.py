import argparse
import calendar
import re

import pandas as pd
from tqdm import tqdm


def main(csv_path, map_path, out_path):
    tqdm.pandas()
    df = pd.read_csv(csv_path, quoting=0, header=None)

    print(f"Replacing deidentification placeholders using {map_path} ...")

    # Placeholder -> Surrogate accoring to external CSV
    df_map = pd.read_csv(map_path, quoting=0, header=None)
    df_map = df_map.fillna("")
    mapping = {row[0]: row[1] for row in df_map.values}

    df[1] = df[1].progress_apply(
        lambda x: "@@@".join(
            [
                mapping[item] if item in mapping.keys() else item
                for item in x.replace("[**", "@@@[**")
                .replace("**]", "**]@@@")
                .split("@@@")
            ]
        ).replace("@@@", "")
    )

    print("Done!")

    # Convert the other placeholders

    def backdate(string):
        """
        Backdate all of the dates in a record by the same interval.
            (e.g., "[**2177-12-5**] to [**2192-1-3**]"
                -> "[**1997-12-5**] to [**2012-1-3**]")
        """
        # Search for [**2000-1-1**] to [**2219-12-31**]
        r_ymd = re.compile(
            r"\[\*\*(22[01][0-9]|2[0-1][0-9][0-9])-([0-9]+)-([0-9]+)\*\*\]"
        )
        placeholders_ymd = sorted(set(r_ymd.findall(string)))[::-1]
        placeholder_dicts_ymd = []
        for ymd in placeholders_ymd:
            placeholder_dicts_ymd.append({"y": ymd[0], "m": ymd[1], "d": ymd[2]})

        # Search for [**1-/2000**] to [**12/2219**]
        r_ym = re.compile(
            r"\[\*\*(1[0-2]-?|-?[1-9]-?)/(22[01][0-9]|2[0-1][0-9][0-9])\*\*\]"
        )
        placeholders_ym = sorted(
            [(ym[1], ym[0]) for ym in list(set(r_ym.findall(string)))]
        )[::-1]
        placeholder_dicts_ym = []
        for ym in placeholders_ym:
            placeholder_dicts_ym.append({"y": ym[0], "m": ym[1]})

        # Search for [**yyyy**]
        r_y = re.compile(r"\[\*\*([12][0-9]{3})\*\*\]")
        placeholders_y = sorted(set(r_y.findall(string)))[::-1]
        placeholder_dicts_y = []
        for y in placeholders_y:
            placeholder_dicts_y.append({"y": y})

        # Search for '[**yy**]
        r_y_short = re.compile(r"'\[\*\*([0-9]+)\*\*\]([^\"])")
        y_short_placeholders = set(r_y_short.findall(string))

        # Search for [**100-1-1**] to [**1999-12-31**] & [**2220-1-1**] to [**9999-12-31**]
        r_ymd_large = re.compile(
            r"\[\*\*(22[2-9][0-9]|2[3-9][0-9]+|[13456789][0-9]+)"
            + r"-([0-9]+)-([0-9]+)\*\*\]"
        )
        placeholders_ymd_outrange = sorted(set(r_ymd_large.findall(string)))[::-1]
        placeholder_dicts_ymd_outrange = []
        for ymd in placeholders_ymd_outrange:
            placeholder_dicts_ymd_outrange.append(
                {"y": ymd[0], "m": ymd[1], "d": ymd[2]}
            )

        # Search for [**1-/300**] to [**12/999**] & [**1-/2220**] to [**12/9999**]
        r_ym_large = re.compile(
            r"\[\*\*(1[0-2]-?|-?[1-9]-?)/"
            + r"(22[2-9][0-9]|2[3-9][0-9]+|[3-9][0-9]+)\*\*\]"
        )
        placeholders_ym_outrange = sorted(
            [(ym[1], ym[0]) for ym in list(set(r_ym_large.findall(string)))]
        )
        placeholder_dicts_ym_outrange = []
        for ym in placeholders_ym_outrange:
            placeholder_dicts_ym_outrange.append({"y": ym[0], "m": ym[1]})

        # Search for [**m-d**]:
        r_md = re.compile(r"\[\*\*([0-9]+-[0-9]+)\*\*\]")
        placeholders_md = set(r_md.findall(string))
        placeholder_dicts_md = []
        for md in placeholders_md:
            placeholder_dicts_md.append({"md": md})

        if len(placeholders_ymd) == 0:
            return string
        else:
            # Determine year subset so that
            # the latest year in the range (2000, 2019)
            y_latest = max(
                int(placeholder_dicts_ymd[0]["y"]),
                int(placeholder_dicts_ym[0]["y"] if placeholder_dicts_ym else 0),
                int(placeholder_dicts_y[0]["y"] if placeholder_dicts_y else 0),
            )
            y_latest_surrogate = 2000 + y_latest % 20
            y_subset = y_latest - y_latest_surrogate

            ymd_map = {
                f'[**{d["y"]}-{d["m"]}-{d["d"]}**]': f'{d["m"]}/{d["d"]}/{int(d["y"])-y_subset}'
                for d in placeholder_dicts_ymd
            }
            ym_map = {
                f'[**{d["m"]}/{d["y"]}**]': f'{d["m"].replace("-", "")}/{int(d["y"])-y_subset}'
                for d in placeholder_dicts_ym
            }
            y_map = {
                f'[**{d["y"]}**]': f'{int(d["y"])-y_subset}'
                for d in placeholder_dicts_y
            }
            y_short_map = {
                f"'[**{ph[0]}**]{ph[1]}": f"'{(int(ph[0])-y_subset)%100:02d}{ph[1]}"
                for ph in y_short_placeholders
            }

            # e.g., [**263-11-28**] -> 11/28
            ymd_outrange_map = {
                f'[**{d["y"]}-{d["m"]}-{d["d"]}**]': f'{d["m"]}/{d["d"]}'
                for d in placeholder_dicts_ymd_outrange
            }

            # e.g., [**10/2570**] -> Oct
            ym_outrange_map = {
                f'[**{d["m"]}/{d["y"]}**]': f'{calendar.month_abbr[int(d["m"].replace("-", ""))]}'
                for d in placeholder_dicts_ym_outrange
            }

            # e.g., [**5-8**]: -> 5/8:
            md_map = {f'[**{d["md"]}**]:': f'{d["md"]}:' for d in placeholder_dicts_md}

            for mapping in [
                ymd_map,
                ym_map,
                y_map,
                y_short_map,
                ymd_outrange_map,
                ym_outrange_map,
                md_map,
            ]:
                if mapping:
                    for k, v in mapping.items():
                        string = string.replace(k, v)
                else:
                    continue

            return string

    print("Backdating noteevents ...")
    df[1] = df[1].progress_map(backdate)
    df.to_csv(out_path, header=False, index=False)
    print(f"Done! -> {out_path}")


if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(dest="input_path", help="path of summary csv file")
        parser.add_argument(
            dest="map_path", help="path of placeholder:surrogate mapping csv"
        )
        parser.add_argument(
            dest="output_path_csv", help="path to save cleaned csv file"
        )
        args = parser.parse_args()
        return args

    tqdm.pandas()
    config = get_args()
    main(config.input_path, config.map_path, config.output_path_csv)

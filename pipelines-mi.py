import argparse
import os
import pathlib
import shutil
import warnings

import pandas as pd
import yaml
from easydict import EasyDict

from wbe_odm import odm, utilities
from wbe_odm.odm_mappers import mi_mapper


def str2bool(arg):
    str_yes = {"y", "yes", "t", "true"}
    str_no = {"n", "no", "f", "false"}
    value = arg.lower()
    if value in str_yes:
        return True
    elif value in str_no:
        return False
    else:
        raise argparse.ArgumentTypeError("Unrecognized boolean value.")


def str2list(arg):
    return arg.lower().split("-")


def load_files_from_folder(folder, extension):
    files = os.listdir(folder)
    return [file for file in files if "$" not in file and extension in file]


def get_data_excerpt(origin_folder):
    short_csv_path = os.path.join(os.path.dirname(origin_folder), "short_csv")
    files = load_files_from_folder(origin_folder, "csv")
    for file in files:
        path = os.path.join(origin_folder, file)
        df = pd.read_csv(path)
        if len(df) > 1000:
            df = df.sample(n=1000)
        df.to_csv(os.path.join(short_csv_path, file), sep=",", index=False)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-con",
        "--config",
        type=str,
        default="pipelines-config-mi.yaml",
        help="Config file where all the paths are defined",
    )  # noqa

    args = parser.parse_args()
    pipeline_config_path = args.config

    with open(pipeline_config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    if not os.path.exists(config.csv_folder):
        raise ValueError("CSV folder does not exist. Please modify config file.")

    warnings.filterwarnings("error")

    store = odm.Odm()

    data_folder = pathlib.Path(config.data_folder)

    source_files = EasyDict(
        {k: str(data_folder / v) for k, v in config.source_files.items()}
    )

    print("Importing data from St. Paul...")
    mapper1 = mi_mapper.MiLabMapper(config_file=config.mapper_config)
    mapper1.read(source_files.lab_data, source_files.static_data)
    store.append_from(mapper1)

    mapper2 = mi_mapper.MiHealthMapper(config_file=config.mapper_config)
    mapper2.read(
        source_files.hosps_data,
        source_files.static_data,
        mi_mapper.MI_HOSP_MAP_FILEPATH,
    )
    store.append_from(mapper2)

    mapper3 = mi_mapper.MiHealthMapper(config_file=config.mapper_config)
    mapper3.read(
        source_files.deaths_data,
        source_files.static_data,
        mi_mapper.MI_DEATHS_MAP_FILEPATH,
    )
    store.append_from(mapper3)

    mapper4 = mi_mapper.MiPctPosMapper(config_file=config.mapper_config)
    mapper4.read(source_files.ppos_data, source_files.static_data)
    store.append_from(mapper4)

    mapper5 = mi_mapper.MiVaccineMapper(config_file=config.mapper_config)
    mapper5.read(source_files.vaccines_data, source_files.static_data)
    store.append_from(mapper5)

    mapper6 = mi_mapper.MiCaseMapper(config_file=config.mapper_config)
    mapper6.read(source_files.cases_data, source_files.static_data)
    store.append_from(mapper6)

    print("Removing older dataset...")
    for root, dirs, files in os.walk(config.csv_folder):
        for f in files:
            os.unlink(os.path.join(str(root), str(f)))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    print("Saving dataset...")
    store.to_csv(config.csv_folder)
    print(f"Saved to folder {config.csv_folder}")

    combined = store.combine_dataset()
    combined = utilities.typecast_wide_table(combined)
    combined_path = os.path.join(config.csv_folder, "_" + "combined.csv")
    combined.to_csv(combined_path, sep=",", index=False)
    print(f"Saved Combined dataset to folder {config.csv_folder}.")

import argparse
import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from easydict import EasyDict

import visualizations
from wbe_odm import odm, utilities
from wbe_odm.odm_mappers import (
    inspq_mapper,
    mcgill_mapper,
    modeleau_mapper,
    parquet_folder_mapper,
    vdq_mapper,
)


def str2bool(arg):
    str_yes = {"y", "yes", "t", "true"}
    str_no = {"n", "no", "f", "false"}
    value = arg.lower()
    if value in str_yes:
        return True
    elif value in str_no:
        return False
    else:
        raise argparse.ArgumentError("Unrecognized boolean value.")


def str2list(arg):
    return arg.lower().split("-")


def load_files_from_folder(folder, extension):
    files = os.listdir(folder)
    return [file for file in files if "$" not in file and extension in file]


def get_data_excerpt(origin_folder):
    short_parquet_path = os.path.join(os.path.dirname(origin_folder), "short_parquet")
    files = load_files_from_folder(origin_folder, "parquet")
    for file in files:
        path = os.path.join(origin_folder, file)
        df = pd.read_parquet(path)
        if len(df) > 1000:
            df = df.sample(n=1000)
        df.to_parquet(os.path.join(short_parquet_path, file), sep=",", index=False)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-scty",
        "--cities",
        type=str2list,
        default="qc-mtl-lvl-gtn",
        help="Cities to load data from",
    )  # noqa
    parser.add_argument(
        "-st",
        "--sitetypes",
        type=str2list,
        default="wwtpmus-wwtpmuc-lagoon",
        help="Types of sites to parse",
    )  # noqa
    parser.add_argument(
        "-cphd",
        "--publichealth",
        type=str2bool,
        default=True,
        help="Include public health data (default=True",
    )  # noqa
    parser.add_argument(
        "-re",
        "--reload",
        type=str2bool,
        default=False,
        help="Reload from raw sources (default=False) instead of from the current parquet",
    )  # noqa
    parser.add_argument(
        "-sh",
        "--short",
        type=str2bool,
        default=False,
        help="Generate a small dataset for testing purposes",
    )  # noqa
    parser.add_argument(
        "-gd",
        "--generate",
        type=str2bool,
        default=False,
        help="Generate datasets for machine learning (default=False)",
    )  # noqa
    parser.add_argument(
        "-dcty",
        "--datacities",
        type=str2list,
        default="qc-mtl-lvl-gtn",
        help="Cities for which to generate datasets for machine learning (default=qc)",
    )  # noqa
    parser.add_argument(
        "-web", "--website", type=str2bool, default=False, help="Build website files."
    )  # noqa
    parser.add_argument(
        "-wcty",
        "--webcities",
        type=str2list,
        default="qc-mtl-lvl-gtn",
        help="Cities to display on the website",
    )  # noqa
    parser.add_argument(
        "-con",
        "--config",
        type=str,
        default="pipelines-config2022.yaml",
        help="Config file where all the paths are defined",
    )  # noqa
    args = parser.parse_args()

    source_cities = args.cities
    sitetypes = args.sitetypes
    publichealth = args.publichealth
    reload = args.reload
    generate = args.generate
    website = args.website
    generate = args.generate
    dataset_cities = args.datacities
    web_cities = args.webcities
    short = args.short
    config = args.config

    if config:
        with open(config, "r") as f:
            config = EasyDict(yaml.safe_load(f))

    if not os.path.exists(config.parquet_folder):
        raise ValueError("parquet folder does not exist. Please modify config file.")

    warnings.filterwarnings("error")

    store = odm.Odm()
    print(source_cities)

    static_path = os.path.join(config.data_folder, config.static_data)
    if reload:
        if "qc" in source_cities:
            print("Importing data from Quebec City...")
            print("Importing viral data from Quebec City...")
            qc_lab = mcgill_mapper.McGillMapper()
            virus_path = os.path.join(config.data_folder, config.qc_virus_data)

            qc_lab.read(
                virus_path, static_path, config.qc_virus_sheet_name, config.qc_virus_lab
            )  # noqa
            print("Adding Quality Checks for Qc...")
            qc_quality_checker = mcgill_mapper.QcChecker()
            qc_lab = qc_quality_checker.read_validation(
                qc_lab, virus_path, config.qc_quality_sheet_name
            )

            store.append_from(qc_lab)

            print("Importing Wastewater lab data from Quebec City...")
            modeleau = modeleau_mapper.ModelEauMapper()
            path = os.path.join(config.data_folder, config.qc_lab_data)
            modeleau.read(
                path,
                config.qc_sheet_name,
                lab_id=config.qc_lab,
                start="2022-03-01",
                end=None,
            )
            store.append_from(modeleau)

            print("Importing Quebec city lab data...")
            subfolder = os.path.join(config.data_folder, config.qc_city_plant_folder)
            files = load_files_from_folder(subfolder, "xls")
            for file in files:
                vdq_plant = vdq_mapper.VdQPlantMapper2022()
                print(f"Parsing file {file}...")
                vdq_plant.read(os.path.join(subfolder, file))
                store.append_from(vdq_plant)

        if "mtl" in source_cities:
            print("Importing viral data from Montreal...")
            mtl_lab = mcgill_mapper.McGillMapper()
            virus_path = os.path.join(config.data_folder, config.mtl_virus_data)

            mtl_lab.read(
                virus_path,
                static_path,
                config.mtl_virus_sheet_name,
                config.mtl_virus_lab,
            )  # noqa
            print("Adding Quality Checks for Montreal...")
            mtl_quality_checker = mcgill_mapper.QcChecker()
            mtl_lab = mtl_quality_checker.read_validation(
                mtl_lab, virus_path, config.mtl_quality_sheet_name
            )

            store.append_from(mtl_lab)

        if "lvl" in source_cities:
            print("Importing viral data from Laval...")
            lvl_lab = mcgill_mapper.McGillMapper()
            virus_path = os.path.join(config.data_folder, config.lvl_virus_data)

            lvl_lab.read(
                virus_path,
                static_path,
                config.lvl_virus_sheet_name,
                config.lvl_virus_lab,
            )  # noqa
            print("Adding Quality Checks for Laval...")
            lvl_quality_checker = mcgill_mapper.QcChecker()
            lvl_lab = lvl_quality_checker.read_validation(
                lvl_lab, virus_path, config.lvl_quality_sheet_name
            )

            store.append_from(lvl_lab)

        if "gtn" in source_cities:
            print("Importing viral data from Gatineau...")
            gtn_lab = mcgill_mapper.McGillMapper()
            virus_path = os.path.join(config.data_folder, config.gtn_virus_data)

            gtn_lab.read(
                virus_path,
                static_path,
                config.gtn_virus_sheet_name,
                config.gtn_virus_lab,
            )  # noqa
            print("Adding Quality Checks for Gatineau...")
            gtn_quality_checker = mcgill_mapper.QcChecker()
            gtn_lab = gtn_quality_checker.read_validation(
                gtn_lab, virus_path, config.gtn_quality_sheet_name
            )

            store.append_from(gtn_lab)

        if publichealth:
            print("Importing case data from INSPQ...")
            public_health = inspq_mapper.INSPQ_mapper()
            path = (
                os.path.join(config.data_folder, config.inspq_data)
                if config.inspq_data
                else None
            )

            public_health.read(path, start="2022-03-01", end=None)
            store.append_from(public_health)
            print("Importing vaccine data from INSPQ...")
            vacc = inspq_mapper.INSPQVaccineMapper()
            path = (
                os.path.join(config.data_folder, config.inspq_vaccine_data)
                if config.inspq_vaccine_data
                else None
            )

            vacc.read(path, start="2022-03-01", end=None)
            store.append_from(vacc)

        print("Removing older dataset...")
        for root, dirs, files in os.walk(config.parquet_folder):
            for f in files:
                os.unlink(os.path.join(str(root), str(f)))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

        print("Saving dataset...")
        store.to_parquet(config.parquet_folder)
        print(f"Saved to folder {config.parquet_folder}")

        if short:
            get_data_excerpt(config.parquet_folder)
        print("Saving combined dataset...")

        combined = store.combine_dataset()
        combined = utilities.typecast_wide_table(combined)
        combined_path = os.path.join(config.parquet_folder, "_" + "combined.parquet")
        combined.to_parquet(combined_path)
        print(f"Saved Combined dataset to folder {config.parquet_folder}.")

    if not reload:
        print("Reading data back from parquet...")
        store = odm.Odm()
        from_parquet = parquet_folder_mapper.ParquetFolderMapper()
        from_parquet.read(config.parquet_folder)
        store.append_from(from_parquet)

        print("Reading combined data back from parquet...")
        for root, dirs, files in os.walk(config.parquet_folder):
            for f in files:
                if "combined" in f:
                    combined_path = str(f)
                    break
        if combined_path is None:
            combined = pd.DataFrame()
        combined = pd.read_parquet(os.path.join(config.parquet_folder, str(f)))
        combined = combined.replace("nan", np.nan)
        combined = utilities.typecast_wide_table(combined)

    if website:
        if "bsl" in web_cities:
            print(f"BSL cities found in config file are {config.bsl_cities}")
            web_cities.remove("bsl")
            web_cities.extend(config.bsl_cities)
        print("Generating website files...")
        labels = visualizations.read_labels()

        sites = store.site
        sites["siteID"] = sites["siteID"].str.lower()
        sites = sites.drop_duplicates(subset=["siteID"], keep="first").copy()

        site_type_filt = sites["type"].str.lower().str.contains("|".join(sitetypes))
        sites = sites.loc[site_type_filt]

        city_filt = sites["siteID"].str.contains("|".join(web_cities))
        sites = sites.loc[city_filt]
        print("building site geojson...")
        visualizations.get_site_geoJSON(
            sites,
            combined,
            labels,
            config.site_output_dir,
            config.site_name,
            config.colors,
            config.default_start_date,
        )
        print("Building polygon geojson...")
        poly_list = sites["polygonID"].to_list()
        visualizations.build_polygon_geoJSON(
            store,
            poly_list,
            config.polygon_output_dir,
            config.poly_name,
            config.polys_to_extract,
        )

        for site_id in sites["siteID"].to_list():
            print("building website plots for ", site_id, "...")
            plot_start_date = config.default_start_date
            plot_data, metadata = visualizations.centreau_website_data(
                combined, labels, site_id, plot_start_date
            )
            if (
                isinstance(plot_data, pd.DataFrame)
                and plot_data.empty
                or not isinstance(plot_data, pd.DataFrame)
                and not plot_data
            ):
                continue
            visualizations.plot_centreau(
                plot_data,
                metadata,
                plot_start_date,
                config.plot_output_dir,
                labels,
                config.logo_path,
                lod=config.lod,
                langs=config.plot_langs,
            )

    if generate:
        date = datetime.now().strftime("%Y-%m-%d")
        print("Generating site-specific wide datasets...")
        if "bsl" in dataset_cities:
            dataset_cities.remove("bsl")
            dataset_cities.extend(config.bsl_cities)
        sites = store.site
        for city in dataset_cities:
            filt_city = sites["siteID"].str.contains(city)
            site_type_filt = sites["type"].str.contains("|".join(sitetypes))
            city_sites = (
                sites.loc[filt_city & site_type_filt, "siteID"].dropna().unique()
            )
            for city_site in city_sites:
                print(f"Generating dataset for {city_site}")
                dataset = utilities.build_site_specific_dataset(combined, city_site)
                dataset = utilities.resample_per_day(dataset)
                path = os.path.join(config.city_output_dir, f"{city_site}.parquet")
                dataset.to_parquet(path)

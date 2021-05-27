import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import *
from wbe_odm import odm, utilities
from wbe_odm.odm_mappers import (
    csv_folder_mapper,
    inspq_mapper,
    mcgill_mapper,
    modeleau_mapper,
    vdq_mapper
)


def str2bool(arg):
    value = arg.lower()
    if value in STR_YES:
        return True
    elif value in STR_NO:
        return False
    else:
        raise argparse.ArgumentError('Unrecognized boolean value.')


def str2list(arg):
    return arg.lower().split("-")


def make_point_feature(row, props_to_add):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [row["geoLong"], row["geoLat"]],
            },
        "properties": {
            k: row[k] for k in props_to_add
        }
    }


def get_latest_sample_date(df):
    if len(df) == 0:
        return pd.NaT
    df = df.sort_values(by="Calculated_timestamp")
    return df.iloc[-1, df.columns.get_loc("Calculated_timestamp")]


def get_cm_to_plot(samples, thresh_n):
    # the type to plot depends on:
    # 1) What is the latest collection method for samples at that site
    # 2) How many samples of that cm there are
    possible_cms = ["ps", "cp", "grb"]
    last_dates = []
    n_samples = []
    for cm in possible_cms:
        samples_of_type = samples.loc[
            samples["Sample_collection"].str.contains(cm)
        ]
        n_samples.append(len(samples_of_type))
        last_dates.append(get_latest_sample_date(samples_of_type))
    series = [pd.Series(x) for x in [possible_cms, n_samples, last_dates]]
    types = pd.concat(series, axis=1)
    types.columns = ["type", "n", "last_date"]
    types = types.sort_values("last_date", ascending=True)

    # if there is no collection method that has enough
    # samples to satisfy the threshold, that condition is moot
    types = types.loc[~types["last_date"].isna()]
    if len(types.loc[types["n"] >= thresh_n]) == 0:
        thresh_n = 0
    types = types.loc[types["n"] >= thresh_n]
    if len(types) == 0:
        return None
    return types.iloc[-1, types.columns.get_loc("type")]


def get_samples_for_site(site_id, df):
    sample_filter1 = df["Sample_siteID"].str.lower() == site_id.lower()
    return df.loc[sample_filter1].copy()


def get_viral_measures(df):
    cols_to_remove = []
    for col in df.columns:
        l_col = col.lower()
        cond1 = "wwmeasure" in l_col
        cond2 = "covn2" in l_col or 'npmmov' in l_col
        cond3 = "gc" in l_col
        if (cond1 and cond2 and cond3) or "timestamp" in l_col:
            continue
        cols_to_remove.append(col)
    df.drop(columns=cols_to_remove, inplace=True)
    return df


def get_site_list(sites):
    return sites["siteID"].dropna().unique().to_list()


def get_last_sunday(date):
    if date is None:
        date = pd.to_datetime("01-01-2020")
    date = date.to_pydatetime()
    offset = (date.weekday() - 6) % 7
    return date - timedelta(days=offset)


def combine_viral_cols(viral):
    sars = []
    pmmov = []
    for col in viral.columns:
        if "timestamp" in col:
            continue
        _, *desc = col.split("_")
        virus, _, _, _ = desc.lower().split("_")
        if "cov" in virus:
            sars.append(col)
        elif "pmmov" in virus:
            pmmov.append(col)
    for name, ls in zip(["sars", "pmmov"], [sars, pmmov]):
        viral[name] = viral[ls].mean(axis=1)
    viral.drop(columns=sars+pmmov, inplace=True)
    return viral


def get_samples_in_interval(samples, dateStart, dateEnd):
    if pd.isna(dateStart) and pd.isna(dateEnd):
        return samples
    elif pd.isna(dateStart):
        return samples.loc[samples["Calculated_timestamp"] <= dateEnd]
    elif pd.isna(dateEnd):
        return samples.loc[samples["Calculated_timestamp"] >= dateStart]
    return samples.loc[
        samples["Calculated_timestamp"] >= dateStart &
        samples["Calculated_timestamp"] <= dateEnd]


def get_samples_to_plot(samples, cm):
    if pd.isna(cm):
        return None
    return samples.loc[
        samples["Sample_collection"].str.contains(cm)]


def get_viral_timeseries(samples):
    viral = get_viral_measures(samples)
    viral = combine_viral_cols(viral)
    viral["norm"] = normalize_by_pmmv(viral)
    return viral


def normalize_by_pmmv(df):
    div = df["sars"] / df["pmmov"]
    div = div.replace([np.inf], np.nan)
    return div[~div.isna()]


def build_empty_color_ts(date_range):
    df = pd.DataFrame(date_range)
    df.columns = ["last_sunday"]
    df["norm"] = np.nan
    return df


def get_n_bins(series, all_colors):
    max_len = len(all_colors)-1
    len_not_null = len(series[~series.isna()])
    if len_not_null == 0:
        return None
    elif len_not_null < max_len:
        return len_not_null
    return max_len


def get_color_ts(samples,
                 colorscale,
                 dateStart=DEFAULT_START_DATE,
                 dateEnd=None):
    dateStart = pd.to_datetime(dateStart)
    weekly = None
    if samples is not None:
        viral = get_viral_timeseries(samples)
        if viral is not None:
            viral["last_sunday"] = viral["Calculated_timestamp"].apply(
                get_last_sunday)
            weekly = viral.resample("W", on="last_sunday").median()

    date_range_start = get_last_sunday(dateStart)
    if dateEnd is None:
        dateEnd = pd.to_datetime("now")
    date_range = pd.date_range(start=date_range_start, end=dateEnd, freq="W")
    result = pd.DataFrame(date_range)
    result.columns = ["date"]
    result.sort_values("date", inplace=True)

    if weekly is None:
        weekly = build_empty_color_ts(date_range)
    weekly.sort_values("last_sunday", inplace=True)
    result = pd.merge(
        result,
        weekly,
        left_on="date",
        right_on="last_sunday",
        how="left")

    n_bins = get_n_bins(result["norm"], colorscale)
    if n_bins is None:
        result["signal_strength"] = 0
    elif n_bins == 1:
        result["signal_strength"] = 1
    else:
        result["signal_strength"] = pd.cut(
            result["norm"],
            n_bins,
            labels=range(1, n_bins+1))
    result["signal_strength"] = result["signal_strength"].astype("str")
    result.loc[result["signal_strength"].isna(), "signal_strength"] = "0"
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.set_index("date", inplace=True)
    return pd.Series(result["signal_strength"]).to_dict()


def get_website_type(types):
    site_types = {
        "wwtpmuc": {
            "french": "StaRRE municipale pour égouts unitaires",  # noqa
            "english": "WRRF for combined sewers"
        },
        "pstat": {
            "french": "Station de pompage",
            "english": "Pumping station"
            },
        "ltcf": {
            "french": "Établissement de soins de longue durée",
            "english": "Long-term care facility"
            },
        "airpln": {
            "french": "Avion",
            "english": "Airplane"
            },        
        "corFcil": {
            "french": "Prison",
            "english": "Correctional facility"
            },
        "school": {
            "french": "École",
            "english": "School"
            },        
        "hosptl": {
            "french": "Hôpital",
            "english": "Hospital"
            },
        "shelter": {
            "french": "Refuge",
            "english": "Shelter"
            },
        "swgTrck": {
            "french": "Camion de vidange",
            "english": "Sewage truck"
            },
        "uCampus": {
            "french": "Campus universitaire",
            "english": "University campus"
            },
        "mSwrPpl": {
            "french": "Collecteur d'égouts",
            "english": "Sewer main collector"
            },
        "holdTnk": {
            "french": "Bassin de stockage",
            "english": "Holding tank"
            },
        "retPond": {
            "french": "Bassin de rétention",
            "english": "Retention tank"
            },
        "wwtpMuS": {
            "french": "StaRRE municipale pour égouts sanitaires",  # noqa
            "english": "Municipal WRRF for sanitary sewers"
            },
        "wwtpInd": {
            "french": "StaRRE eaux industrielles",
            "english": "WWRF for industrial waters"
            },
        "lagoon": {
            "french": "Étang aéré",
            "english": "Aerated lagoon"
            },
        "septTnk": {
            "french": "Fosse septique",
            "english": "Septic tank"
            },
        "river": {
            "french": "Rivière",
            "english": "River"
            },
        "lake": {
            "french": "Lac",
            "english": "Lake",
        },
        "estuary": {
            "french": "Estuaire",
            "english": "Estuary"
            },
        "sea": {
            "french": "Mer",
            "english": "Sea",
            },
        "ocean": {
            "french": "Océan",
            "english": "Ocean"
            },
    }
    return types.str.lower().map(site_types)


def get_website_name(names):
    sitename_lang_map = {
        "quebec station est":{
            "french": "Québec Station Est",
            "english": "Québec East WRRF",
        },
        "quebec station ouest":{
            "french": "Québec Station Ouest",
            "english": "Québec West WRRF",
        },
        "montréal intercepteur nord":{
            "french": "Montréal Intercepteur Nord",
            "english": "Montreal North Intercepter",
        },
        "montréal intercepteur sud":{
            "french": "Montréal Intercepteur Sud",
            "english": "Montreal South Intercepter",
        },
        "station rimouski": {
            "french": "StaRRE de Rimouski",
            "english": "Rimouski WRRF",
        },
        "station rivière-du-loup": {
            "french": "StaRRE de Rivière-du-Loup",
            "english": "Rivière-du-Loup WRRF",
        },
        "station st-alexandre-de-kamouraska": {
            "french": "StaRRE de St-Alexandre-de Kamouraska",
            "english": "St-Alexandre-de-Kamouraska WRRF",
        },
        "trois-pistoles": {
            "french": "StaRRE de Trois-Pistoles",
            "english": "Trois-Pistoles WRRF",
        },
        "matane": {
            "french": "StaRRE de Matane",
            "english": "Matane WRRF",
        },
        "auteuil": {
            "french": "StaRRE Auteuil",
            "english": "Auteuil WRRF",
        },
        "fabreville": {
            "french": "StaRRE Fabreville",
            "english": "Fabreville WRRF",
        },
        "station de pompage sainte-dorothée": {
            "french": "Station de pompage de Ste-Dorothée",
            "english": "Ste-Dorothée pumping station",
        },
        "station de pompage bertrand": {
            "french": "Station de pompage Bertrand",
            "english": "Bertrand pumping station",
        },
        "la pinière": {
            "french": "StaRRE de La Pinière",
            "english": "La Pinière WRRF",
        },
    }
    return names.map(sitename_lang_map)


def get_municipality(ids):
    municipalities = {
        "qc": "Québec",
        "mtl": "Montréal",
        "lvl": "Laval",
        "tr": "Trois-Rivières",
        "dr": "Drummondville",
        "vc": "Victoriaville",
        "riki": "Rimouski",
        "rdl": "Rivière-du-Loup",
        "stak": "Saint-Alexandre-de-Kamouraska",
        "3p": "Trois-Pistoles",
        "mtn": "Matane"
    }
    city_id = ids.str.lower().apply(lambda x: x.split("_")[0])
    return city_id.map(municipalities)


def website_collection_method(cm):
    collection = {
        "cp": {
            "french": "Composite",
            "english": "Composite"},
        "grb": {
            "french": "Ponctuel",
            "english": "Grab"},
        "ps": {
            "french": "Passif",
            "english": "Passive"
        }
    }
    return cm.map(collection)


def get_site_geoJSON(
        sites,
        combined,
        site_output_dir,
        site_name,
        colorscale,
        dateStart=None,
        dateEnd=None,):

    sites["samples_for_site"] = sites.apply(
        lambda row: get_samples_for_site(row["siteID"], combined),
        axis=1)
    sites["samples_in_range"] = sites.apply(
        lambda row: get_samples_in_interval(
            row["samples_for_site"], dateStart, dateEnd),
        axis=1)
    sites["collection_method"] = sites.apply(
        lambda row: get_cm_to_plot(
            row["samples_in_range"], thresh_n=7),
        axis=1)
    sites["samples_to_plot"] = sites.apply(
        lambda row: get_samples_to_plot(
            row["samples_in_range"], row["collection_method"]),
        axis=1)
    sites["date_color"] = sites.apply(
        lambda row: get_color_ts(
            row["samples_to_plot"], colorscale, dateStart, dateEnd),
        axis=1)

    sites["clean_type"] = get_website_type(sites["type"])
    sites["municipality"] = get_municipality(sites["siteID"])
    sites["name"] = get_website_name(sites["name"])

    sites["collection_method"] = website_collection_method(
        sites["collection_method"])
    cols_to_keep = [
        "siteID",
        "name",
        "description",
        "clean_type",
        "polygonID",
        "municipality",
        "collection_method",
        "date_color"]
    sites.fillna("", inplace=True)
    sites["features"] = sites.apply(
        lambda row: make_point_feature(row, cols_to_keep), axis=1)
    point_list = list(sites["features"])
    js = {
        "type": "FeatureCollection",
        "features": point_list,
        "colorKey": colorscale
    }
    path = os.path.join(site_output_dir, site_name)
    with open(path, "w") as f:
        f.write(json.dumps(js, indent=4))
    return


def build_polygon_geoJSON(store, poly_list, output_dir, name, types=None):
    polys = store.get_polygon_geoJSON(types=types)
    features = polys["features"]
    for feature in features.copy():
        props = feature["properties"]
        poly_id = props["polygonID"]
        if poly_id not in poly_list:
            features.remove(feature)
    polys["feature"] = features
    path = os.path.join(output_dir, name)
    with open(path, "w") as f:
        f.write(json.dumps(polys, indent=4))


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
        df.to_csv(
            os.path.join(short_csv_path, file),
            sep=",", index=False)


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-scty', '--cities', type=str2list, default="qc-mtl-lvl-bsl", help='Cities to load data from')  # noqa
    parser.add_argument('-st', '--sitetypes', type=str2list, default="wwtpmus-wwtpmuc-lagoon", help='Types of sites to parse')  # noqa
    parser.add_argument('-cphd', '--publichealth', type=str2bool, default=True, help='Include public health data (default=True')  # noqa
    parser.add_argument('-re', '--reload', type=str2bool, default=False, help='Reload from raw sources (default=False) instead of from the current csv')  # noqa
    parser.add_argument('-sh', '--short', type=str2bool, default=False, help='Generate a small dataset for testing purposes')  # noqa
    parser.add_argument('-gd', '--generate', type=str2bool, default=False, help='Generate datasets for machine learning (default=False)')  # noqa
    parser.add_argument('-dcty', '--datacities', type=str2list, default="qc", help='Cities for which to generate datasets for machine learning (default=qc)')  # noqa
    
    parser.add_argument('-web', '--website', type=str2bool, default=False, help='build geojson files for website (default=False)')  # noqa
    args = parser.parse_args()


    source_cities = args.cities
    sitetypes = args.sitetypes
    publichealth = args.publichealth
    generate = args.generate
    website = args.website
    reload = args.reload
    generate = args.generate
    dataset_cities = args.datacities
    short = args.short

    if not os.path.exists(CSV_FOLDER):
        raise ValueError(
            "CSV folder does not exist. Please modify config file.")

    store = odm.Odm()
    print(source_cities)
    if reload:
        if "qc" in source_cities:
            print("Importing data from Quebec City...")
            print("Importing viral data from Quebec City...")
            qc_lab = mcgill_mapper.McGillMapper()
            qc_lab.read(QC_VIRUS_DATA, STATIC_DATA, QC_VIRUS_SHEET_NAME, QC_VIRUS_LAB)  # noqa
            quality_checker = mcgill_mapper.QcChecker()
            qc_lab = quality_checker.read_validation(qc_lab, QC_VIRUS_DATA, QC_QUALITY_SHEET_NAME)
            store.append_from(qc_lab)
            print("Importing Wastewater lab data from Quebec City...")
            modeleau = modeleau_mapper.ModelEauMapper()
            modeleau.read(QC_LAB_DATA, QC_SHEET_NAME, lab_id=QC_LAB)
            store.append_from(modeleau)
            print("Importing Quebec city sensor data...")
            subfolder = os.path.join(
                os.path.join(DATA_FOLDER, QC_CITY_SENSOR_FOLDER))
            files = load_files_from_folder(subfolder, "xls")
            for file in files:
                vdq_sensors = vdq_mapper.VdQSensorsMapper()
                print("Parsing file " + file + "...")
                vdq_sensors.read(os.path.join(subfolder, file))
                store.append_from(vdq_sensors)
            print("Importing Quebec city lab data...")
            subfolder = os.path.join(DATA_FOLDER, QC_CITY_PLANT_FOLDER)
            files = load_files_from_folder(subfolder, "xls")
            for file in files:
                vdq_plant = vdq_mapper.VdQPlantMapper()
                print("Parsing file " + file + "...")
                vdq_plant.read(os.path.join(subfolder, file))
                store.append_from(vdq_plant)

        if "mtl" in source_cities:
            print("Importing data from Montreal...")
            mcgill_lab = mcgill_mapper.McGillMapper()
            poly_lab = mcgill_mapper.McGillMapper()
            print("Importing viral data from McGill...")
            mcgill_lab.read(MTL_LAB_DATA, STATIC_DATA, MTL_MCGILL_SHEET_NAME, MCGILL_VIRUS_LAB)  # noqa
            print("Importing viral data from Poly...")
            poly_lab.read(MTL_LAB_DATA, STATIC_DATA, MTL_POLY_SHEET_NAME, POLY_VIRUS_LAB)  # noqa
            store.append_from(mcgill_lab)
            store.append_from(poly_lab)
        
        if "bsl" in source_cities:
            print(f"BSL cities found in config file are {BSL_CITIES}")
            source_cities.remove("bsl")
            source_cities.extend(BSL_CITIES)
            print("Importing data from Bas St-Laurent...")
            bsl_lab = mcgill_mapper.McGillMapper()
            bsl_lab.read(BSL_LAB_DATA, STATIC_DATA, BSL_SHEET_NAME, BSL_VIRUS_LAB)  # noqa
            store.append_from(bsl_lab)

        if "lvl" in source_cities:
            print("Importing data from Laval...")
            lvl_lab = mcgill_mapper.McGillMapper()
            lvl_lab.read(LVL_LAB_DATA, STATIC_DATA, LVL_SHEET_NAME, LVL_VIRUS_LAB)  # noqa
            store.append_from(lvl_lab)

        if publichealth:
            print("Importing case data from INSPQ...")
            public_health = inspq_mapper.INSPQ_mapper()
            public_health.read(INSPQ_DATA)
            store.append_from(public_health)

        print("Removing older dataset...")
        for root, dirs, files in os.walk(CSV_FOLDER):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

        print("Saving dataset...")
        prefix = datetime.now().strftime("%Y-%m-%d")
        store.to_csv(CSV_FOLDER, prefix)
        print(f"Saved to folder {CSV_FOLDER} with prefix \"{prefix}\"")

        if short:
            get_data_excerpt(CSV_FOLDER)
        print("Saving combined dataset...")
        combined = store.combine_dataset()
        combined = utilities.typecast_wide_table(combined)
        combined_path = os.path.join(CSV_FOLDER, prefix+"_"+"combined.csv")
        combined.to_csv(combined_path, sep=",", index=False)
        print(f"Saved Combined dataset to folder {CSV_FOLDER}.")

    if not reload:
        print("Reading data back from csv...")
        store = odm.Odm()
        from_csv = csv_folder_mapper.CsvFolderMapper()
        from_csv.read(CSV_FOLDER)
        store.append_from(from_csv)

        print("Reading combined data back from csv...")
        for root, dirs, files in os.walk(CSV_FOLDER):
            for f in files:
                if "combined" in f:
                    combined_path = f
                    break
        if combined_path is None:
            combined = pd.DataFrame()
        combined = pd.read_csv(os.path.join(CSV_FOLDER, f), low_memory=False)
        combined = utilities.typecast_wide_table(combined)

    if website:
        print("Generating website files...")
        sites = store.site
        sites["siteID"] = sites["siteID"].str.lower()
        sites = sites.drop_duplicates(subset=["siteID"], keep="first").copy()

        site_type_filt = sites["type"].str.lower().str.contains('|'.join(sitetypes))
        sites = sites.loc[site_type_filt]

        city_filt = sites["siteID"].str.contains('|'.join(source_cities))
        sites = sites.loc[city_filt]

        get_site_geoJSON(
            sites,
            combined,
            SITE_OUTPUT_DIR,
            SITE_NAME,
            COLORS,
            dateStart=None,
            dateEnd=None)

        poly_list = sites["polygonID"].to_list()
        build_polygon_geoJSON(
            store, poly_list, POLYGON_OUTPUT_DIR, POLY_NAME, POLYS_TO_EXTRACT)

    if generate:
        date = datetime.now().strftime("%Y-%m-%d")
        print("Generating ML Dataset...")
        sites = store.site
        for city in dataset_cities:
            filt_city = sites["siteID"].str.contains(city)
            site_type_filt = sites["type"].str.contains('|'.join(sitetypes))
            city_sites = sites.loc[filt_city & site_type_filt, "siteID"].dropna().unique()
            for city_site in city_sites:
                print(f"Generating dataset for {city_site}")
                dataset = utilities.build_site_specific_dataset(combined, city_site)
                dataset = utilities.resample_per_day(dataset)
                dataset = dataset["2021-02-01":]
                dataset.to_csv(os.path.join(CITY_OUTPUT_DIR, f"{city_site}.csv"))

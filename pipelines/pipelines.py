import argparse
import base64
import json
import os
import shutil
import yaml
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from easydict import EasyDict
from plotly.express import colors as pc
from plotly.subplots import make_subplots

# from config import *
from wbe_odm import odm, utilities
from wbe_odm.odm_mappers import (csv_folder_mapper, inspq_mapper,
                                 mcgill_mapper, modeleau_mapper, vdq_mapper)


def str2bool(arg):
    str_yes = {'y', 'yes', 't', 'true'}
    str_no = {'n', 'no', 'f', 'false'}
    value = arg.lower()
    if value in str_yes:
        return True
    elif value in str_no:
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
    df = df.sort_index()
    return df.iloc[-1].name


def get_cm_to_plot(samples, thresh_n):
    if isinstance(samples, pd.DataFrame):
        if samples.empty:
            return None
    elif not samples:
        return None
    # the type to plot depends on:
    # 1) What is the latest collection method for samples at that site
    # 2) How many samples of that cm there are
    possible_cms = ["ps", "cp", "grb"]
    last_dates = []
    n_samples = []
    for cm in possible_cms:
        samples_of_type = samples.loc[
            samples["Sample_collection"].str.contains(cm, na=False)
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


def get_site_list(sites):
    return sites["siteID"].dropna().unique().to_list()


def get_last_sunday(date):
    if pd.isna(date):
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
        table, virus, unit, agg, var = col.lower().split("_")

        if "cov" in virus:
            sars.append(col)
        elif "pmmov" in virus:
            pmmov.append(col)
    viral.drop(columns=sars+pmmov, inplace=True)
    return viral


def get_samples_in_interval(samples, dateStart, dateEnd):
    if pd.isna(dateStart) and pd.isna(dateEnd):
        return samples
    elif pd.isna(dateStart):
        return samples.loc[: dateEnd]
    elif pd.isna(dateEnd):
        return samples.loc[dateStart:]
    return samples.loc[dateStart: dateEnd]


def get_samples_of_collection_method(samples, cm):
    if pd.isna(cm):
        return None
    return samples.loc[
        samples["Sample_collection"].str.contains(cm, na=False)]


def get_viral_timeseries(samples):
    if isinstance(samples, pd.DataFrame):
        if samples.empty:
            return None
    elif not samples:
        return None
    table = "WWMeasure"
    unit = 'gcml'
    agg_method = 'single-to-mean'
    value_cols = []
    dfs = []
    covn2_col = None
    for virus in ['npmmov', 'covn2']:
        common = "_".join([table, virus, unit, agg_method])
        value_col = "_".join([common, 'value'])
        value_cols.append(value_col)
        if 'covn2' in value_col:
            covn2_col = value_col
        elif 'npmmov' in value_col:
            npmmov_col = value_col
        quality_col = "_".join([common, 'qualityFlag'])
        df = samples.loc[:, [value_col, quality_col]]
        quality_filt = ~df[quality_col].str.lower().str.contains('true')
        df = df.loc[quality_filt]
        dfs.append(df)

    viral = pd.concat(dfs, axis=1)
    viral = viral[[col for col in viral.columns if 'value' in col]]
    viral["norm"] = viral[covn2_col] / viral[npmmov_col]
    return viral


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


def get_color_ts(viral,
                 colorscale,
                 dateStart="2021-01-01",
                 dateEnd=None):
    dateStart = pd.to_datetime(dateStart)
    weekly = None
    if viral is not None:
        viral["last_sunday"] = viral.index.map(get_last_sunday)
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
        "corfcil": {
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
        "swgtrck": {
            "french": "Camion de vidange",
            "english": "Sewage truck"
            },
        "ucampus": {
            "french": "Campus universitaire",
            "english": "University campus"
            },
        "mswrppl": {
            "french": "Collecteur d'égouts",
            "english": "Sewer main collector"
            },
        "holdtnk": {
            "french": "Bassin de stockage",
            "english": "Holding tank"
            },
        "retpond": {
            "french": "Bassin de rétention",
            "english": "Retention tank"
            },
        "wwtpmus": {
            "french": "StaRRE municipale pour égouts sanitaires",  # noqa
            "english": "Municipal WRRF for sanitary sewers"
            },
        "wwtpind": {
            "french": "StaRRE eaux industrielles",
            "english": "WWRF for industrial waters"
            },
        "lagoon": {
            "french": "Étang aéré",
            "english": "Aerated lagoon"
            },
        "septtnk": {
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


sitename_lang_map = {
        "québec station est": {
            "french": "Québec Station Est",
            "english": "Québec East WRRF",
        },
        "québec station ouest": {
            "french": "Québec Station Ouest",
            "english": "Québec West WRRF",
        },
        "montréal intercepteur nord": {
            "french": "Montréal Intercepteur Nord",
            "english": "Montreal North Intercepter",
        },
        "montréal intercepteur sud": {
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


def get_website_name(name):
    return sitename_lang_map[name]


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
        "trpis": "Trois-Pistoles",
        "mtne": "Matane"
    }


def get_municipality(id):
    city_id = str(id).lower().split("_")[0]
    return municipalities[city_id]


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


def website_collection_method(cm):
    return collection.get(cm, "")


poly_names = {
    "qc_01_swrcat": {
        'french': "Bassin versant de la station Québec Est",
        "english": "Sewershed of Québec East WRRF",
    },
    "qc_02_swrcat": {
        'french': "Bassin Versant de la station Québec Ouest",
        "english": "Sewershed of Québec West WRRF",
    },
    "prov_qc_hlthreg_laval": {
        'french': "Laval",
        "english": "Laval",
    },
    "prov_qc_hlthreg_chaudiere-appalaches": {
        'french': "Chaudière-Appalaches",
        "english": "Chaudière-Appalaches",
    },
    "prov_qc_hlthreg_nord-du-quebec": {
        'french': "Nord-du-Québec",
        "english": "Nord-du=Québec",
    },
    "prov_qc_hlthreg_estrie": {
        'french': "Estrie",
        "english": "Estrie",
    },
    "prov_qc_hlthreg_capitale-nationale": {
        'french': "Capitale-Nationale",
        "english": "Capitale-Nationale",
    },
    "prov_qc_hlthreg_mauricie-centre-du-quebec": {
        'french': "Mauricie - Centre-du-Québec",
        "english": "Mauricie - Centre-du-Québec",
    },
    "prov_qc_hlthreg_monteregie": {
        'french': "Montérégie",
        "english": "Montérégie",
    },
    "prov_qc_hlthreg_abitibi-temiscamingue": {
        'french': "Abitibi - Témiscamingue",
        "english": "Abitibi - Témiscamingue",
    },
    "prov_qc_hlthreg_outaouais": {
        'french': "Outaouais",
        "english": "Outaouais",
    },
    "prov_qc_hlthreg_bas-saint-laurent": {
        'french': "Bas-Saint-Laurent",
        "english": "Bas-Saint-Laurent",
    },
    "prov_qc_hlthreg_cote-nord": {
        'french': "Côte-Nord",
        "english": "Côte-Nord",
    },
    "prov_qc_hlthreg_montreal": {
        'french': "Montréal",
        "english": "Montréal",
    },
    "prov_qc_hlthreg_laurentides": {
        'french': "Laurentides",
        "english": "Laurentides",
    },
    "prov_qc_hlthreg_lanaudiere": {
        'french': "Lanaudière",
        "english": "Lanaudière",
    },
    "prov_qc_hlthreg_saguenay-lac-saint-jean": {
        'french': "Saguenay - Lac-Saint-Jean",
        "english": "Saguenay - Lac-Saint-Jean",
    },
    "prov_qc_hlthreg_gaspesie-iles-de-la-madeleine": {
        'french': "Gaspésie - Îles-de-la-Madeleine",
        "english": "Gaspésie - Îles-de-la-Madeleine",
    },
    "mtl_01_swrcat": {
        'french': "Bassin versant de l'intercepteur Nord de Montréal",
        "english": "Montréal North Intercepter Sewer Catchment",
    },
    "mtl_02_swrcat": {
        'french': "Bassin versant de l'intercepteur Sud de Montréal",
        "english": "Montréal South Intercepter Sewer Catchment",
    },
    "riki_01_swrcat": {
        'french': "Bassin versant des égouts de Rimouski",
        "english": "Rimouski WRRF sewershed",
    },
    "mtne_01_swrcat": {
        'french': "Bassin versant des égouts de Matane",
        "english": "Matane WRRF sewershed",
    },
    "trpis_01_swrcat": {
        'french': "Bassin versant des égouts de Trois-Pistoles",
        "english": "Trois-Pistoles WRRF sewershed",
    },
    "rdl_01_swrcat": {
        'french': "Bassin versant des égouts de Rivière-du-Loup",
        "english": "Rivière-du-Loup WRRF sewershed",
    },
    "stak_01_swrcat": {
        'french': "Bassin versant des égouts de St-Alexandre-de-Kamouraska",
        "english": "St-Alexandre-de-Kamouraska WRRF sewershed",
    },
    "lvl_01_swrcat": {
        'french': "Bassin versant des égouts de la Station Auteuil",
        "english": "Auteuil WRRF sewershed",
    },
    "lvl_02_swrcat": {
        'french': "Bassin versant des égouts de la station Fabreville",
        "english": "Fabreville WRRF sewershed",
    },
    "lvl_03_swrcat": {
        'french': "Bassin versant de Ste-Dorothée",
        "english": "Ste-Dorothée sewershed",
    },
    "LVL_04_swrCat": {
        'french': "Bassin versant des égouts de Bertrand",
        "english": "Bertand sewershed",
    },
    "lvl_05_swrcat": {
        'french': "Bassin versant des égouts de la Pinière",
        "english": "La Pinière sewershed",
    },
}


def clean_polygon_name(poly_id):
    return poly_names[poly_id]


def get_samples_to_plot(site_dataset, dateStart=None, dateEnd=None):
    samples_in_range = get_samples_in_interval(
        site_dataset, dateStart, dateEnd)
    collection_method = get_cm_to_plot(samples_in_range, thresh_n=7)
    return get_samples_of_collection_method(
        samples_in_range, collection_method)


def get_site_geoJSON(
        sites,
        combined,
        site_output_dir,
        site_name,
        colorscale,
        dateStart,
        dateEnd=None):

    sites["dataset"] = sites.apply(
        lambda row: utilities.build_site_specific_dataset(
            combined, row["siteID"]),
        axis=1)
    sites["dataset"] = sites.apply(
        lambda row: utilities.resample_per_day(row['dataset']),
        axis=1)
    sites['samples'] = sites.apply(
        lambda row: get_samples_to_plot(row['dataset'], dateStart, dateEnd),
        axis=1)
    sites["viral"] = sites.apply(
        lambda row: get_viral_timeseries(row['samples']),
        axis=1)
    sites["date_color"] = sites.apply(
        lambda row: get_color_ts(
            row["viral"], colorscale, dateStart, dateEnd),
        axis=1)

    sites["clean_type"] = get_website_type(sites["type"])
    sites["municipality"] = sites['siteID'].apply(
        lambda x: get_municipality(x))
    sites["name"] = sites['name'].apply(
        lambda x: get_website_name(x))
    sites['collection_method'] = sites.apply(
        lambda row: get_cm_to_plot(row['samples'], thresh_n=7), axis=1)
    sites["collection_method"] = sites["collection_method"].apply(
        lambda x: website_collection_method(x))
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


def get_info_from_col(col, df):
    found = df[col].value_counts().index.to_list()
    if found:
        return found[0]
    return None


def centreau_website_data(combined, site_id, dateStart, dateEnd=None):
    site_dataset = utilities.build_site_specific_dataset(combined, site_id)
    site_dataset = utilities.resample_per_day(site_dataset)
    samples = get_samples_to_plot(site_dataset, dateStart, dateEnd)
    viral = get_viral_timeseries(samples)
    if (
        isinstance(viral, pd.DataFrame)
        and viral.empty
        or not isinstance(viral, pd.DataFrame)
        and not viral
    ):
        return None, None
    sars_col = [col for col in viral.columns if 'covn2' in col][0]
    pmmv_col = [col for col in viral.columns if 'npmmov' in col][0]
    norm_col = 'norm'
    cases_col = 'CPHD_conf_report_value'

    poly_id = get_info_from_col('CPHD-Polygon_polygonID', samples)
    site_name = get_info_from_col('Site_name', samples)

    df = pd.concat([viral, site_dataset[cases_col]], axis=1)
    df = df[dateStart:]

    df.rename(columns={
        sars_col: 'sars',
        pmmv_col: 'pmmv',
        norm_col: 'norm',
        cases_col: 'cases',
    }, inplace=True)
    metadata = {
        'poly_name': clean_polygon_name(poly_id),
        'site_id': site_id,
        'site_name': get_website_name(site_name),
    }
    return df, metadata


def get_plot_titles(metadata):
    return {
        'french': f'Surveillance SRAS-CoV-2 via les eaux usées<br>{metadata["site_name"]["french"]}',  # noqa
        'english': f'SARS-CoV-2 surveilance in wastewater<br>{metadata["site_name"]["english"]}'  # noqa
    }


def get_axes_titles():
    return {
        1: {
            'french': 'Nouveaux cas',
            'english': 'New cases',
        },
        2: {
            'french': 'Signal viral normalisé (gc/gc)',
            'english': 'Normalized viral signal (gc/gc)'
        },
        3: {
            'french': 'Signal viral (gc/ml)',
            'english': 'Viral signal (gc/ml)',
        }
    }


def get_column_names(metadata):
    return {
        'sars': {
            'french': 'SRAS (gc/ml)',
            'english': 'SARS (gc/ml)'
        },
        'pmmv': {
            'french': 'PMMoV (gc/ml)',
            'english': 'PMMoV (gc/ml)'
        },
        'norm': {
            'french': 'SRAS/PMMoV (gc/gc)',
            'english': 'SARS/PMMoV (gc/gc)'
        },
        'cases': {
            "french": f'Cas journaliers {metadata["poly_name"]["french"]}',
            "english": f'Daily cases {metadata["poly_name"]["english"]}',
        }
    }


def update_webplot_layout(fig, x0, lang, plot_titles, axes_titles):
    fig.update_layout(
        xaxis_title="Date",
        xaxis_tick0=x0,
        xaxis_dtick=14 * 24 * 3600000,
        xaxis_tickformat="%d-%m-%Y",
        xaxis_tickangle=30, plot_bgcolor="white",
        xaxis_gridcolor="rgba(100,100,100,0.10)",
        yaxis_gridcolor="rgba(0,0,0,0)",
        xaxis_ticks="outside",

        hovermode='x unified',  # To compare on hover
        title=plot_titles[lang],
        legend=dict(
            yanchor="top",
            xanchor="left",
            orientation="h",
            y=1.05,
            x=0
        ),
        xaxis=dict(
            domain=[0.12, 1]
        ),
        yaxis=dict(
            title=axes_titles[1][lang],
            side="right",
            domain=[0, 0.9],
        ),
        yaxis2=dict(
            title=dict(
                text=axes_titles[2][lang],
                standoff=0.01,
            ),
            side="left",
            anchor="x",
        ),
        yaxis3=dict(
            title=dict(
                text=axes_titles[3][lang],
                standoff=0.01,
            ),
            overlaying="y",
            side="left",
            position=0
        ),
    )
    return fig


def add_logo_to_plot(fig, path):
    encoded_image = base64.b64encode(open(path, 'rb').read())
    fig.add_layout_image(
        dict(
            source='data:image/png;base64,{}'.format(encoded_image.decode()),
            xref="paper", yref="paper",
            x=1.125, y=1.00,
            sizex=0.5, sizey=0.25,
            xanchor="right", yanchor="bottom"
            )
    )
    return fig


def plot_web(data,
             metadata,
             dateStart,
             output_dir,
             lod=0,
             langs=['french', 'english']):
    # sourcery no-metrics
    plot_titles = get_plot_titles(metadata)
    axes_titles = get_axes_titles()
    col_names = get_column_names(metadata)
    first_sunday = get_last_sunday(pd.to_datetime(dateStart))
    for lang in langs:
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"secondary_y": True}]])
        colors = pc.qualitative.Plotly
        line_colors = [color for i, color in enumerate(colors) if i != 2]
        bar_color = colors[2]

        for i, col in enumerate(
                col for col in data.columns if 'case' not in col):
            marker_color = line_colors[i]
            if 'norm' not in col:
                marker_colors = data[col].apply(
                    lambda x: utilities.hex_color_adder(
                        marker_color, '#7d7d7d')
                    if x < lod else marker_color
                )
            else:
                marker_colors = marker_color
            trace = go.Scatter(
                x=data.index,
                y=data[col],
                name=col_names[col][lang],
                mode="lines+markers",
                marker=dict(color=marker_colors),
                connectgaps=True,
                visible='legendonly' if 'sars' not in col else True,
                yaxis="y3" if 'norm' not in col else "y2",
                hovertemplate=' %{y:.3f}'
            )
            fig.add_trace(trace)

        cases_trace = go.Bar(
            x=data.index,
            y=data['cases'],
            name=col_names['cases'][lang],
            marker=dict(
                opacity=0.3,
                color=bar_color
            ),
            hovertemplate=' %{y}<extra>Nouveaux cas</extra>'
        )
        fig.add_trace(cases_trace)

        fig = update_webplot_layout(fig,
                                    first_sunday,
                                    lang,
                                    plot_titles,
                                    axes_titles)

        fig = add_logo_to_plot(fig, config.logo_path)
        if langs == ['french']:
            fig.write_html(f"{output_dir}/{metadata['site_id']}.html")
        else:
            fig.write_html(f"{output_dir}/{metadata['site_id']}_{lang}.html")
    return


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-scty', '--cities', type=str2list, default="qc-mtl-lvl-bsl", help='Cities to load data from')  # noqa
    parser.add_argument('-st', '--sitetypes', type=str2list, default="wwtpmus-wwtpmuc-lagoon", help='Types of sites to parse')  # noqa
    parser.add_argument('-cphd', '--publichealth', type=str2bool, default=True, help='Include public health data (default=True')  # noqa
    parser.add_argument('-re', '--reload', type=str2bool, default=False, help='Reload from raw sources (default=False) instead of from the current csv')  # noqa
    parser.add_argument('-sh', '--short', type=str2bool, default=False, help='Generate a small dataset for testing purposes')  # noqa
    parser.add_argument('-gd', '--generate', type=str2bool, default=False, help='Generate datasets for machine learning (default=False)')  # noqa
    parser.add_argument('-dcty', '--datacities', type=str2list, default="qc-mtl-lvl-bsl", help='Cities for which to generate datasets for machine learning (default=qc)')  # noqa
    parser.add_argument('-web', '--website', type=str2bool, default=False, help="Build website files.")  # noqa
    parser.add_argument('-wcty', '--webcities', type=str2list, default="qc-mtl-lvl-bsl", help='Cities to display on the website')  # noqa
    parser.add_argument('-con', '--config', type=str, default='pipelines.yaml', help="Config file where all the paths are defined")  # noqa
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

    if not os.path.exists(config.csv_folder):
        raise ValueError(
            "CSV folder does not exist. Please modify config file.")

    store = odm.Odm()
    print(source_cities)

    static_path = os.path.join(config.data_folder, config.static_data)
    if reload:
        if "qc" in source_cities:
            print("Importing data from Quebec City...")
            print("Importing viral data from Quebec City...")
            qc_lab = mcgill_mapper.McGillMapper()
            virus_path = os.path.join(config.data_folder, config.qc_virus_data)

            qc_lab.read(virus_path, static_path, config.qc_virus_sheet_name, config.qc_virus_lab)  # noqa
            print("Adding Quality Checks for Qc...")
            qc_quality_checker = mcgill_mapper.QcChecker()
            qc_lab = qc_quality_checker.read_validation(
                qc_lab,
                virus_path,
                config.qc_quality_sheet_name)

            store.append_from(qc_lab)
            print("Importing Wastewater lab data from Quebec City...")
            modeleau = modeleau_mapper.ModelEauMapper()
            path = os.path.join(config.data_folder, config.qc_lab_data)
            modeleau.read(path, config.qc_sheet_name, lab_id=config.qc_lab)
            store.append_from(modeleau)
            print("Importing Quebec city sensor data...")
            subfolder = os.path.join(
                os.path.join(config.data_folder, config.qc_city_sensor_folder))
            files = load_files_from_folder(subfolder, "xls")
            for file in files:
                vdq_sensors = vdq_mapper.VdQSensorsMapper()
                print("Parsing file " + file + "...")
                vdq_sensors.read(os.path.join(subfolder, file))
                store.append_from(vdq_sensors)
            print("Importing Quebec city lab data...")
            subfolder = os.path.join(
                config.data_folder,
                config.qc_city_plant_folder)
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
            virus_path = os.path.join(config.data_folder, config.mtl_lab_data)
            mcgill_lab.read(virus_path, static_path, config.mtl_mcgill_sheet_name, config.mcgill_virus_lab)  # noqa
            print("Importing viral data from Poly...")
            poly_lab.read(virus_path, static_path, config.mtl_poly_sheet_name, config.poly_virus_lab)  # noqa
            print("Adding Quality Checks for mtl...")
            mtl_quality_checker = mcgill_mapper.QcChecker()

            store.append_from(mcgill_lab)
            store.append_from(poly_lab)
            store = mtl_quality_checker.read_validation(
                store,
                virus_path,
                config.mtl_quality_sheet_name)

        if "bsl" in source_cities:
            print(f"BSL cities found in config file are {config.bsl_cities}")
            source_cities.remove("bsl")
            source_cities.extend(config.bsl_cities)
            print("Importing data from Bas St-Laurent...")
            bsl_lab = mcgill_mapper.McGillMapper()
            virus_path = os.path.join(config.data_folder, config.bsl_lab_data)
            bsl_lab.read(virus_path, static_path, config.bsl_sheet_name, config.bsl_virus_lab)  # noqa
            print("Adding Quality Checks for BSL...")
            bsl_quality_check = mcgill_mapper.QcChecker()
            bsl_quality_check.read_validation(
                bsl_lab,
                virus_path,
                config.bsl_quality_sheet_name)
            store.append_from(bsl_lab)

        if "lvl" in source_cities:
            print("Importing data from Laval...")
            lvl_lab = mcgill_mapper.McGillMapper()
            virus_path = os.path.join(config.data_folder, config.lvl_lab_data)
            lvl_lab.read(
                virus_path,
                static_path,
                config.lvl_sheet_name,
                config.lvl_virus_lab)  # noqa
            print("Adding Quality Checks for Laval...")
            lvl_quality_checker = mcgill_mapper.QcChecker()
            lvl_quality_checker.read_validation(
                lvl_lab,
                virus_path,
                config.lvl_quality_sheet_name)
            store.append_from(lvl_lab)

        if publichealth:
            print("Importing case data from INSPQ...")
            public_health = inspq_mapper.INSPQ_mapper()
            if not config.inspq_data:
                path = None
            else:
                path = os.path.join(config.data_folder, config.inspq_data)
            public_health.read(path)
            store.append_from(public_health)
            print("Importing vaccine data from INSPQ...")
            vacc = inspq_mapper.INSPQVaccineMapper()
            if not config.inspq_vaccine_data:
                path = None
            else:
                path = os.path.join(
                    config.data_folder,
                    config.inspq_vaccine_data)
            vacc.read(path)
            store.append_from(vacc)

        print("Removing older dataset...")
        for root, dirs, files in os.walk(config.csv_folder):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

        print("Saving dataset...")
        store.to_csv(config.csv_folder)
        print(f"Saved to folder {config.csv_folder}")

        if short:
            get_data_excerpt(config.csv_folder)
        print("Saving combined dataset...")

        combined = store.combine_dataset()
        combined = utilities.typecast_wide_table(combined)
        combined_path = os.path.join(config.csv_folder, "_"+"combined.csv")
        combined.to_csv(combined_path, sep=",", index=False)
        print(f"Saved Combined dataset to folder {config.csv_folder}.")

    if not reload:
        print("Reading data back from csv...")
        store = odm.Odm()
        from_csv = csv_folder_mapper.CsvFolderMapper()
        from_csv.read(config.csv_folder)
        store.append_from(from_csv)

        print("Reading combined data back from csv...")
        for root, dirs, files in os.walk(config.csv_folder):
            for f in files:
                if "combined" in f:
                    combined_path = f
                    break
        if combined_path is None:
            combined = pd.DataFrame()
        combined = pd.read_csv(
            os.path.join(config.csv_folder, f), low_memory=False)
        combined = combined.replace('nan', np.nan)
        combined = utilities.typecast_wide_table(combined)

    if website:
        if "bsl" in web_cities:
            print(f"BSL cities found in config file are {config.bsl_cities}")
            web_cities.remove("bsl")
            web_cities.extend(config.bsl_cities)
        print("Generating website files...")
        sites = store.site
        sites["siteID"] = sites["siteID"].str.lower()
        sites = sites.drop_duplicates(subset=["siteID"], keep="first").copy()

        site_type_filt = sites["type"]\
            .str.lower().str.contains('|'.join(sitetypes))
        sites = sites.loc[site_type_filt]

        city_filt = sites["siteID"].str.contains('|'.join(web_cities))
        sites = sites.loc[city_filt]
        print("building site geojson...")
        get_site_geoJSON(
            sites,
            combined,
            config.site_output_dir,
            config.site_name,
            config.colors,
            config.default_start_date)
        print("Building polygon geojson...")
        poly_list = sites["polygonID"].to_list()
        build_polygon_geoJSON(
            store,
            poly_list,
            config.polygon_output_dir,
            config.poly_name,
            config.polys_to_extract)

        for site_id in sites['siteID'].to_list():
            print("building website plots for ", site_id, "...")
            plot_start_date = config.lvl_start_date\
                if 'lvl' in site_id.lower()\
                else config.default_start_date
            plot_data, metadata = centreau_website_data(
                combined,
                site_id,
                plot_start_date)
            if (
                isinstance(plot_data, pd.DataFrame)
                and plot_data.empty
                or not isinstance(plot_data, pd.DataFrame)
                and not plot_data
            ):
                continue
            plot_web(
                plot_data,
                metadata,
                plot_start_date,
                config.plot_output_dir,
                od=config.lod,
                langs=config.plot_langs)

    if generate:
        date = datetime.now().strftime("%Y-%m-%d")
        print("Generating ML Dataset...")
        if "bsl" in dataset_cities:
            print(f"BSL cities found in config file are {config.bsl_cities}")
            dataset_cities.remove("bsl")
            dataset_cities.extend(config.bsl_cities)
        sites = store.site
        for city in dataset_cities:
            filt_city = sites["siteID"].str.contains(city)
            site_type_filt = sites["type"].str.contains('|'.join(sitetypes))
            city_sites = sites.loc[
                    filt_city & site_type_filt, "siteID"]\
                .dropna().unique()
            for city_site in city_sites:
                print(f"Generating dataset for {city_site}")
                dataset = utilities.build_site_specific_dataset(
                    combined, city_site)
                dataset = utilities.resample_per_day(dataset)
                # dataset = dataset["2021-01-01":]
                dataset.to_csv(
                    os.path.join(config.city_output_dir, f"{city_site}.csv"))

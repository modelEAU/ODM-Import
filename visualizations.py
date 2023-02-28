import base64
import copy
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.express import colors as pc
from plotly.subplots import make_subplots

from wbe_odm import utilities


def make_point_feature(row, props_to_add):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [row["geoLong"], row["geoLat"]]},
        "properties": {k: row[k] for k in props_to_add},
    }


def get_latest_sample_date(df):
    if len(df) == 0:
        return pd.NaT
    df = df.sort_index()
    return df.iloc[-1].name


def get_cm_to_plot(samples, thresh_n):
    if (
        isinstance(samples, pd.DataFrame)
        and samples.empty
        or not isinstance(samples, pd.DataFrame)
        and not samples
    ):
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
        _, virus, _, _, _ = col.lower().split("_")

        if "cov" in virus:
            sars.append(col)
        elif "pmmov" in virus:
            pmmov.append(col)
    viral.drop(columns=sars + pmmov, inplace=True)
    return viral


def get_samples_in_interval(samples, dateStart, dateEnd):
    if pd.isna(dateStart) and pd.isna(dateEnd):
        return samples
    elif pd.isna(dateStart):
        return samples.loc[:dateEnd]
    elif pd.isna(dateEnd):
        return samples.loc[dateStart:]
    return samples.loc[dateStart:dateEnd]


def get_samples_of_collection_method(samples, cm):
    if pd.isna(cm):
        return None
    return samples.loc[samples["Sample_collection"].str.contains(cm, na=False)]


def get_viral_timeseries(samples):
    if isinstance(samples, pd.DataFrame):
        if samples.empty:
            return None
    elif not samples:
        return None
    table = "WWMeasure"
    unit = "gcml"
    agg_method = "single-to-mean"
    value_cols = []
    dfs = []
    covn1_col = ""
    pmmov_col = ""
    for virus in ["pmmov", "covn1"]:
        common = "_".join([table, virus, unit, agg_method])
        value_col = "_".join([common, "value"])
        value_cols.append(value_col)
        if "covn1" in value_col:
            covn1_col = value_col
        elif "pmmov" in value_col:
            pmmov_col = value_col
        quality_col = "_".join([common, "qualityFlag"])
        df = samples.loc[:, [value_col, quality_col]]
        quality_filt = ~df[quality_col].str.lower().str.contains("true")
        df = df.loc[quality_filt]
        dfs.append(df)
    if not all([covn1_col, pmmov_col]):
        raise ValueError("Could not find all columns")

    viral = pd.concat(dfs, axis=1)
    viral_columns = viral.columns.to_list()
    value_columns = [col for col in viral_columns if "value" in col]
    viral = viral[value_columns]

    viral["norm"] = viral[covn1_col] / viral[pmmov_col]
    return viral


def build_empty_color_ts(date_range):
    df = pd.DataFrame(date_range)
    df.columns = ["last_sunday"]
    df["norm"] = np.nan
    return df


def get_n_bins(series, all_colors):
    max_len = len(all_colors) - 1
    len_not_null = len(series[~series.isna()])
    if len_not_null == 0:
        return None
    elif len_not_null < max_len:
        return len_not_null
    return max_len


def get_color_ts(viral, colorscale, dateStart="2021-01-01", dateEnd=None):
    dateStart = pd.to_datetime(dateStart, infer_datetime_format=True)
    weekly = None
    if viral is not None:
        viral["last_sunday"] = viral.index.map(get_last_sunday)
        weekly = viral.resample("W", on="last_sunday").median()

    date_range_start = get_last_sunday(dateStart)
    if dateEnd is None:
        dateEnd = pd.to_datetime(datetime.now())
    date_range = pd.date_range(start=date_range_start, end=dateEnd, freq="W")
    result = pd.DataFrame(date_range)
    result.columns = ["date"]
    result.sort_values("date", inplace=True)

    if weekly is None:
        weekly = build_empty_color_ts(date_range)
    weekly.sort_values("last_sunday", inplace=True)
    result = pd.merge(
        result, weekly, left_on="date", right_on="last_sunday", how="left"
    )

    n_bins = get_n_bins(result["norm"], colorscale)
    if n_bins is None:
        result["signal_strength"] = 0
    elif n_bins == 1:
        result["signal_strength"] = 1
    else:
        result["signal_strength"] = pd.cut(
            result["norm"], n_bins, labels=range(1, n_bins + 1)
        )
    result["signal_strength"] = result["signal_strength"].astype("str")
    result.loc[result["signal_strength"].isna(), "signal_strength"] = "0"
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.set_index("date", inplace=True)
    return pd.Series(result["signal_strength"]).to_dict()


def get_website_type(labels, types):
    return types.str.lower().map(labels)


def get_website_name(labels, name):
    return labels[name]


def get_municipality(labels, id):
    city_id = str(id).lower().split("_")[0]
    return labels[city_id]


def website_collection_method(labels, cm):
    return labels.get(cm, "")


def clean_polygon_name(labels, poly_id):
    return labels[poly_id]


def get_samples_to_plot(site_dataset, dateStart=None, dateEnd=None):
    samples_in_range = get_samples_in_interval(site_dataset, dateStart, dateEnd)
    collection_method = get_cm_to_plot(samples_in_range, thresh_n=7)
    return get_samples_of_collection_method(samples_in_range, collection_method)


def get_site_geoJSON(
    sites,
    combined,
    labels,
    site_output_dir,
    site_name,
    colorscale,
    dateStart,
    dateEnd=None,
):
    sites["dataset"] = sites.apply(
        lambda row: utilities.build_site_specific_dataset(combined, row["siteID"]),
        axis=1,
    )
    sites["dataset"] = sites.apply(
        lambda row: utilities.resample_per_day(row["dataset"]), axis=1
    )
    sites["samples"] = sites.apply(
        lambda row: get_samples_to_plot(row["dataset"], dateStart, dateEnd), axis=1
    )
    sites["viral"] = sites.apply(
        lambda row: get_viral_timeseries(row["samples"]), axis=1
    )
    sites["date_color"] = sites.apply(
        lambda row: get_color_ts(row["viral"], colorscale, dateStart, dateEnd), axis=1
    )

    sites["clean_type"] = get_website_type(labels["site_types"], sites["type"])
    sites["municipality"] = sites["siteID"].apply(
        lambda x: get_municipality(labels["municipalities"], x)
    )
    sites["name"] = sites["name"].apply(
        lambda x: get_website_name(labels["site_names"], x)
    )
    sites["collection_method"] = sites.apply(
        lambda row: get_cm_to_plot(row["samples"], thresh_n=7), axis=1
    )
    sites["collection_method"] = sites["collection_method"].apply(
        lambda x: website_collection_method(labels["collection"], x)
    )
    cols_to_keep = [
        "siteID",
        "name",
        "description",
        "clean_type",
        "polygonID",
        "municipality",
        "collection_method",
        "date_color",
    ]
    sites.fillna("", inplace=True)
    sites["features"] = sites.apply(
        lambda row: make_point_feature(row, cols_to_keep), axis=1
    )
    point_list = list(sites["features"])
    js = {"type": "FeatureCollection", "features": point_list, "colorKey": colorscale}
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


def centreau_website_data(
    combined, labels, site_id, health_polygons, dateStart, dateEnd=None
):
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
    viral_columns = viral.columns.to_list()
    sars_col = [col for col in viral_columns if "covn1" in col][0]
    pmmv_col = [col for col in viral_columns if "pmmov" in col][0]
    norm_col = "norm"

    poly_id = health_polygons[site_id]
    cases_col = f"CPHD-{poly_id}_conf_report_value"

    site_name = get_info_from_col("Site_name", samples)

    df = pd.concat([viral, site_dataset[cases_col]], axis=1)
    df = df[dateStart:]

    df.rename(
        columns={
            sars_col: "sars",
            pmmv_col: "pmmv",
            norm_col: "norm",
            cases_col: "cases",
        },
        inplace=True,
    )
    metadata = {
        "poly_name": clean_polygon_name(labels["poly_names"], poly_id),
        "site_id": site_id,
        "site_name": get_website_name(labels["site_names"], site_name),
    }
    return df, metadata


def get_plot_titles(metadata, labels):
    labels = copy.deepcopy(labels)
    for lang in ["french", "english"]:
        labels[lang] = f"{labels[lang]}<br>{metadata['site_name'][lang]}"
    return labels


def get_column_names(labels, metadata):
    labels = copy.deepcopy(labels)
    for lang in ["french", "english"]:
        labels["cases"][lang] = (
            labels["cases"][lang] + " " + metadata["poly_name"][lang]
        )
    return labels


def update_webplot_layout(fig, x0, lang, plot_titles, axes_titles):
    fig.update_layout(
        xaxis_title="Date",
        xaxis_tick0=x0,
        xaxis_dtick=14 * 24 * 3600000,
        xaxis_tickformat="%d-%m-%Y",
        xaxis_tickangle=30,
        plot_bgcolor="white",
        xaxis_gridcolor="rgba(100,100,100,0.10)",
        yaxis_gridcolor="rgba(0,0,0,0)",
        xaxis_ticks="outside",
        hovermode="x unified",  # To compare on hover
        title=plot_titles[lang],
        legend=dict(yanchor="top", xanchor="left", orientation="h", y=1.05, x=0),
        xaxis=dict(domain=[0.12, 1]),
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
            position=0,
        ),
    )
    return fig


def add_logo_to_plot(fig, path):
    with open(path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image.decode()}",
            xref="paper",
            yref="paper",
            x=1.125,
            y=1.00,
            sizex=0.5,
            sizey=0.25,
            xanchor="right",
            yanchor="bottom",
        )
    )
    return fig


def plot_centreau(
    data, metadata, dateStart, output_dir, labels, logo_path, lod=0, langs=None
):
    if langs is None:
        langs = ["french", "english"]
    # sourcery no-metrics
    plot_titles = get_plot_titles(metadata, labels["plot_titles"]["centreau"])
    axes_titles = labels["axis_titles"]["centreau"]
    col_names = get_column_names(labels["variables"], metadata)
    first_sunday = get_last_sunday(
        pd.to_datetime(dateStart, infer_datetime_format=True)
    )
    for lang in langs:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        colors = pc.qualitative.Plotly
        line_colors = [color for i, color in enumerate(colors) if i != 2]
        bar_color = colors[2]

        for i, col in enumerate(col for col in data.columns if "case" not in col):
            marker_color = line_colors[i]
            if "norm" not in col:
                marker_colors = data[col].apply(
                    lambda x: utilities.hex_color_adder(marker_color, "#7d7d7d")
                    if x < lod
                    else marker_color
                )
            else:
                marker_colors = marker_color
            trace = go.Scatter(
                x=data.index,
                y=data[col],
                name=col_names[col][lang],
                mode="lines+markers",
                marker=dict(color=marker_colors),
                connectgaps=False,
                visible="legendonly" if "sars" not in col else True,
                yaxis="y3" if "norm" not in col else "y2",
                hovertemplate=" %{y:.3f}",
            )
            fig.add_trace(trace)

        cases_trace = go.Bar(
            x=data.index,
            y=data["cases"],
            name=col_names["cases"][lang],
            marker=dict(opacity=0.3, color=bar_color),
            hovertemplate=" %{y}<extra>Nouveaux cas</extra>",
        )
        fig.add_trace(cases_trace)

        fig = update_webplot_layout(fig, first_sunday, lang, plot_titles, axes_titles)

        fig = add_logo_to_plot(fig, logo_path)
        if langs == ["french"]:
            fig.write_html(f"{output_dir}/{metadata['site_id']}.html")
        else:
            fig.write_html(f"{output_dir}/{metadata['site_id']}_{lang}.html")
    return


def get_info_from_col(col, df):
    return found[0] if (found := df[col].value_counts().index.to_list()) else None


def read_labels(path="labels.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

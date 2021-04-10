import json

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from wbe_odm import odm
from wbe_odm.odm_mappers import excel_template_mapper, mcgill_mapper


def load_from_mcgill(
        path_to_lab_data,
        path_to_static_data,
        path_to_mapping,
        sheet_name,
        lab_id,
        start_date=None,
        end_date=None):
    mapper = mcgill_mapper.McGillMapper()
    mapper.read(
        path_to_lab_data,
        path_to_static_data,
        path_to_mapping,
        sheet_name,
        lab_id,
        startdate=start_date,
        enddate=end_date
    )
    odm_data = odm.Odm()
    odm_data.load_from(mapper)
    static_data = excel_template_mapper.ExcelTemplateMapper()
    static_data.read(path_to_static_data)
    odm_data.append_from(static_data)
    return odm_data


def get_available_sites(odm_data):
    samples = odm_data.sample
    available_sites = list(samples["siteID"].unique())
    site_dico = {}
    for i, site in enumerate(available_sites):
        if site == "":
            continue
        site_dico[i] = site
    return site_dico


def apply_quality_flag(odm_data, sample_ids, meas_types):
    """Puts the quality flag to True for the given measurement types"""
    measures = odm_data.ww_measure
    sample_filt = (measures["sampleID"].isin(sample_ids))
    type_filt = (measures["type"].isin(meas_types))
    measures.loc[sample_filt & type_filt, "qualityFlag"] = True
    setattr(odm_data, "ww_measure", measures)
    return odm_data


def sample_time(row):
    comp_start = row["dateTimeStart"]
    grab = row["dateTime"]
    if pd.isna(comp_start):
        return grab
    else:
        return comp_start


def date_from_sample_id(samples, sample_id):
    sample_slice = samples.loc[samples["sampleID"] == sample_id]
    try:
        reportDate = sample_slice.iloc[0]["reportDate"]
        return reportDate
    except IndexError:
        return pd.NaT


def get_normalized_data(data, site_id):
    sars_col = ""
    pmmov_col = ""
    if sars_col not in data.columns:
        print(f"no sars measurements found for site {site_id}")
        return None
    data["normalized"] = data[sars_col] / data[pmmov_col]
    data = data.dropna(subset=["normalized"])
    data = data.sort_values("Sample.dateTimeEnd")
    data = data.reset_index()
    return data


def prettify_name(name):
    name_lst = name.split(" ")
    name_lst = [x.title() for x in name_lst]
    name = " ".join(name_lst)
    name = name.replace("Quebec", "Québec")
    name = name.replace("Montreal", "Montréal")
    return name


def graph_normalized(ww_sample, region_name, site_list, max_normalized):
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]])
    traces = []
    for site in site_list:
        name = get_site_name(ww_sample, site)
        site_measures = get_normalized_data(ww_sample, site)
        if site_measures is None:
            print(site)
            continue
        trace = go.Scatter(
            x=site_measures["sampleDate"],
            y=site_measures["normalized"],
            name=prettify_name(name),
            mode="lines+markers",
            text=site_measures["sampleID"],
            hoverinfo="text"
        )
        traces.append(trace)
    for trace in traces:
        fig.add_trace(trace, secondary_y=True)
    fig.update_layout(
        xaxis_title="Date",
        xaxis_tick0="2020-12-27",
        xaxis_dtick=7 * 24 * 3600000,
        xaxis_tickformat="%d-%m-%Y",
        xaxis_tickangle=30, plot_bgcolor="white",
        xaxis_gridcolor="rgba(100,100,100,0.10)",
        yaxis_gridcolor="rgba(0,0,0,0)",
        xaxis_ticks="outside"
    )
    fig.update_yaxes(
        title="SARS-CoV-2 / PMMoV",
        secondary_y=True,
        range=[0, max_normalized])
    if region_name == "Capitale-Nationale":
        region_name = "Région de Québec"
    fig.update_layout(title=dict(
        text=f"Surveillance SRAS-CoV-2 via les eaux usées<br>{region_name}"
        ))
    return fig


def get_cases_from_ledevoir(region_name, start_date=None, end_date=None):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    response = requests.get(
        "https://ledevoir-coronavirus.herokuapp.com/api/v2/reports/ca/qc"
    )
    j = json.loads(response.text)
    region_codes = {}
    for i in range(len(j["regions"])):
        region_codes[j["regions"][i]["name"]] = i
    region_code = region_codes[region_name]
    cases = pd.DataFrame(j["regions"][region_code]["data"])
    cases["date"] = pd.to_datetime(cases["date"])
    cases = cases.set_index("date")
    if pd.isna(start_date) and pd.isna(end_date):
        pass
    elif pd.isna(end_date):
        cases = cases.loc[start_date:]
    elif pd.isna(start_date):
        cases = cases.loc[:end_date]
    else:
        cases = cases.loc[start_date: end_date]
    return cases


def create_graph(data, region, sites, max_norm=None):
    graph = graph_normalized(data, region, sites, max_norm)
    cases = get_cases_from_ledevoir(region, start_date="2021-01-01")
    cases_trace = go.Bar(
        x=cases.index,
        y=cases["dc"],
        name="Nouveaux cas<br>journaliers",
        marker=dict(opacity=0.3)
    )
    graph.add_trace(cases_trace, secondary_y=False)
    graph.update_layout(legend=dict(
        yanchor="top",
        xanchor="left",
        orientation="h",
        y=1.1, x=0
    ))
    graph.update_yaxes(title="Nouveaux cas", side="right", secondary_y=False)
    graph.update_yaxes(side="left", secondary_y=True)
    graph.add_layout_image(
        dict(
            source="https://www.centreau.ulaval.ca/fileadmin/Documents/Image_de_marque/102378_MODIF_LOGO-CENTREAU_noir.jpg",  # noqa
            xref="paper", yref="paper",
            x=1, y=1.00,
            sizex=0.25, sizey=0.25,
            xanchor="right", yanchor="bottom"
        )
    )
    graph.show()
    return graph


def get_site_name(ww_sample, site_id):
    site_rows = ww_sample.loc[ww_sample["Sample.siteID"].str.lower() == str(site_id).lower()]
    if len(site_rows) > 1:
        return ""
    else:
        first_row = site_rows.iloc[0]
        return first_row.loc["Site.name"]

def combine_ww_measure_and_sample(
    ww: pd.DataFrame,
    sample: pd.DataFrame
        ) -> pd.DataFrame:
    """Merges tables on sampleID

    Parameters
    ----------
    ww : pd.DataFrame
        WWMeasure table re-organized by sample
    sample : pd.DataFrame
        The sample table

    Returns
    -------
    pd.DataFrame
        A combined table containing the data from both DataFrames
    """
    return pd.merge(
        sample, ww,
        how="left",
        left_on="Sample.sampleID",
        right_on="WWMeasure.sampleID")

def combine_site_sample(
    sample: pd.DataFrame,
    site: pd.DataFrame
        ) -> pd.DataFrame:
    """Combines the sample table with site-specific data.

    Parameters
    ----------
    sample : pd.DataFrame
        The sample table
    site : pd.DataFrame
        The site table

    Returns
    -------
    pd.DataFrame
        A combined DataFrame joined on siteID
    """
    return pd.merge(
        sample,
        site,
        how="left",
        left_on="Sample.siteID",
        right_on="Site.siteID")



lab_data = "/Users/jeandavidt/Desktop/latest-data/CentrEau-COVID_Resultats_Quebec_final.xlsx"  # noqa
static_data = "/Users/jeandavidt/Desktop/latest-data/Ville de Quebec - All data - v1.1.xlsx"  # noqa
mapping = "/Users/jeandavidt/dev/jeandavidt/ODM Import/Data/Lab/McGill/Final/mcgill_map.csv"  # noqa
sheet_name = "QC Data Daily Samples (McGill)"
lab_id = "modeleau_lab"
start_date = "2021-01-01"
end_date = None
qc_lab_data = load_from_mcgill(
    lab_data,
    static_data,
    mapping,
    sheet_name,
    lab_id,
    start_date=start_date,
    end_date=end_date
    )
bad_samples = [
    "qc_01_cptp24h_pstgrit_2021-02-12_1",
    "qc_01_cptp24h_pstgrit_2021-02-13_1",
    "qc_01_cptp24h_pstgrit_2021-02-14_1",
    "qc_01_cptp24h_pstgrit_2021-02-19_1",
    "qc_01_cptp24h_pstgrit_2021-02-20_1",
    "qc_01_cptp24h_pstgrit_2021-02-21_1",
    "qc_01_cptp24h_pstgrit_2021-02-22_1",
    "qc_01_cptp24h_pstgrit_2021-02-23_1",
    "qc_01_cptp24h_pstgrit_2021-03-01_1",
    "qc_01_cptp24h_pstgrit_2021-03-23_1",
    "qc_02_cptp24h_pstgrit_2021-02-12_1",
    "qc_02_cptp24h_pstgrit_2021-02-13_1",
    "qc_02_cptp24h_pstgrit_2021-02-14_1",
    "qc_02_cptp24h_pstgrit_2021-02-15_1",
    "qc_02_cptp24h_pstgrit_2021-02-16_1",
    "qc_02_cptp24h_pstgrit_2021-02-19_1",
    "qc_02_cptp24h_pstgrit_2021-02-20_1",
    "qc_02_cptp24h_pstgrit_2021-02-21_1",
    "qc_02_cptp24h_pstgrit_2021-02-22_1",
    "qc_02_cptp24h_pstgrit_2021-02-23_1",
    "qc_02_cptp24h_pstgrit_2021-03-23_1",
    "qc_02_cptp24h_pstgrit_2021-03-24_1",
]
qc_lab_data = apply_quality_flag(qc_lab_data, bad_samples, ["covN2", "nPMMoV"])
region = "Capitale-Nationale"
sites = ["qc_01", "qc_02"]
max_norm = 0.2

ww_measure = qc_lab_data.__parse_ww_measure()
ww_measure = qc_lab_data.agg_ww_measure_per_sample(ww_measure)

sample = qc_lab_data.__parse_sample()
merged = qc_lab_data.combine_ww_measure_and_sample(ww_measure, sample)

site = qc_lab_data.__parse_site()
merged = combine_site_sample(merged, site)
combined = combine_ww_measure_and_sample(ww_measure, sample)
combined = combine_site_sample(
            combined, qc_lab_data.site)

qc_graph = create_graph(combined, region, sites, max_norm)

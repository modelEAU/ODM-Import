import os
COLORS = {
    0: {
        "french": "Pas de données",
        "english": "No Data",
        "color": "#ececec"},
    1: {
        "french": "Très faible",
        "english": "Very Low",
        "color": "#6da06f"},
    2: {
        "french": "Faible",
        "english": "Low",
        "color": "#b6e9d1"},
    3: {
        "french": "Moyennement élevé",
        "english": "Somewhat high",
        "color": "#ffbb43"},
    4: {
        "french": "Élevé",
        "english": "High",
        "color": "#ff8652"},
    5: {
        "french": "Très élevé",
        "english": "Very high",
        "color": "#c13525"},
}

LOGO_PATH = "/Users/jeandavidt/dev/jeandavidt/ODM Import/Data/images/graph_logos.png"
STR_YES = [
    'y',
    'yes',
    't',
    'true'
]

STR_NO = [
    'n',
    'no',
    'f',
    'false'
]

DEFAULT_START_DATE = "2021-01-01"

DATA_FOLDER = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/Input"  # noqa
CSV_FOLDER = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/odm_csv"  # noqa
STATIC_DATA = os.path.join(DATA_FOLDER, "CentrEAU-COVID_Static_Data.xlsx")  # noqa

INSPQ_DATA = os.path.join(DATA_FOLDER, "INSPQ/covid19-hist.csv")

QC_LAB_DATA = os.path.join(DATA_FOLDER, "COVIDProject_Lab Measurements.xlsx")  # noqa
QC_SHEET_NAME = "Lab analyses"
QC_LAB = "modeleau_lab"

QC_CITY_SENSOR_FOLDER = "Qc_sensors"
QC_CITY_PLANT_FOLDER = "Qc_plant"
QC_CITY_RAIN_FOLDER = "Qc_rain"

QC_VIRUS_DATA = os.path.join(DATA_FOLDER, "CentrEau-COVID_Resultats_Quebec_final.xlsx")  # noqa
QC_VIRUS_SHEET_NAME = "QC Data Daily Samples (McGill)"
QC_QUALITY_SHEET_NAME = "QC_Compil_STEP (int)"
QC_VIRUS_LAB = "frigon_lab"

MTL_LAB_DATA = os.path.join(DATA_FOLDER, "CentrEau-COVID_Resultats_Montreal_final.xlsx")  # noqa
MTL_POLY_SHEET_NAME = "Mtl Data Daily Samples (Poly)"
MTL_MCGILL_SHEET_NAME = "Mtl Data Daily Samples (McGill)"
MCGILL_VIRUS_LAB = "frigon_lab"
POLY_VIRUS_LAB = "dorner_lab"

BSL_LAB_DATA = os.path.join(DATA_FOLDER, "CentrEau-COVID_Resultats_BSL_final.xlsx")  # noqa
BSL_SHEET_NAME = "BSL Data Daily Samples (UQAR)"
BSL_VIRUS_LAB = "bsl_lab"
BSL_CITIES = ["stak", "3p", "mtne", "riki", "rdl"]

LVL_LAB_DATA = os.path.join(DATA_FOLDER, "CentrEau-COVID_Resultats_Laval_final.xlsx")
LVL_SHEET_NAME = "LVL_Data Daily Samples (Poly)"
LVL_VIRUS_LAB = "dorner_lab"

POLYGON_OUTPUT_DIR = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Website geo"  # noqa
POLY_NAME = "polygons.geojson"
POLYS_TO_EXTRACT = ["swrCat"]

SITE_OUTPUT_DIR = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Website geo"  # noqa
SITE_NAME = "sites.geojson"

CITY_OUTPUT_DIR = "/Users/jeandavidt/OneDrive - Université Laval/COVID/ML_Cities"  # noqa

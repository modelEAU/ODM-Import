import datetime
default_site_measurement = {
    "siteMeasureID": None,
    "siteID": None,
    "instrumentID": None,
    "reporterID": "NielsNicolai",
    "dateTime": None,
    "type": None,
    "aggregation": "single",
    "aggregationDesc": None,
    "value": None,
    "unit": None,
    "accessToPublic": "YES",
    "accessToAllOrg": "YES",
    "accessToPHAC": "YES",
    "accessToLocalHA": "YES",
    "accessToProvHA": "YES",
    "accessToOtherProv": "YES",
    "accessToDetails": "YES",
    "notes": None
}
def build_id(site, type, date):
    date = datetime.strftime(date, format="%yyyy-%mm-%dd")
    return f"{site}_{type}_{date}"
"Quebec_Est_WWTP_envRnF_2020-06-21",
"Quebec_Est_WWTP",
"NielsNicolai",
"2020-06-21 0:00",
"envRnF",
"single",
"Cumulative rainfall in one day.",
"0",
"mm",
"YES",
"YES",
"YES",
"YES",
"YES",
"YES",
"YES",
"retour épaississeurs chargé (+de 1000 mg/l MES)"
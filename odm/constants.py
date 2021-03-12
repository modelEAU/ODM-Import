TYPES = {
    'WWMeasure': {
        'WWMeasureID': 'string',
        'reporterID': 'string',
        'sampleID': 'string',
        'labID': 'string',
        'assayMethodID': 'float64',
        'analysisDate': 'datetime64[ns]',
        'reportDate': 'datetime64[ns]',
        'fractionAnalyzed': 'category',
        'type': 'category',
        'value': 'float64',
        'unit': 'category',
        'aggregation': 'category',
        'index': 'float64',
        'qualityFlag': 'bool',
        'accessToPublic': 'bool',
        'accessToAllOrg': 'bool',
        'accessToPHAC': 'bool',
        'accessToLocalHA': 'bool',
        'accessToProvHA': 'bool',
        'accessToOtherProv': 'bool',
        'accessToDetails': 'bool',
        'notes': 'string'
    },
    'SiteMeasure': {
        'siteMeasureID': 'string',
        'siteID': 'string',
        'instrumentID': 'string',
        'reporterID': 'string',
        'dateTime': 'datetime64[ns]',
        'type': 'category',
        'aggregation': 'category',
        'aggregationDesc': 'string',
        'value': 'float64',
        'unit': 'category',
        'accessToPublic': 'bool',
        'accessToAllOrg': 'bool',
        'accessToPHAC': 'bool',
        'accessToLocalHA': 'bool',
        'accessToProvHA': 'bool',
        'accessToOtherProv': 'bool',
        'accessToDetails': 'bool',
        'notes': 'string'
    },
    'Sample': {
        'sampleID': 'string',
        'siteID': 'string',
        'reporterID': 'string',
        'dateTime': 'datetime64[ns]',
        'dateTimeStart': 'datetime64[ns]',
        'dateTimeEnd': 'datetime64[ns]',
        'type': 'category',
        'collection': 'category',
        'preTreatment': 'string',
        'pooled': 'bool',
        'children': 'string',
        'parent': 'string',
        'sizeL': 'float64',
        'fieldSampleTempC': 'float64',
        'shippedOnIce': 'bool',
        'storageTempC': 'float64',
        'qualityFlag': 'bool',
        'notes': 'string'
    },
    'Site': {
        'siteID': 'string',
        'name': 'string',
        'description': 'string',
        'type': 'string',
        'geoLat': 'float64',
        'geoLong': 'float64',
        'polygonID': 'string',
        'link': 'string',
        'notes': 'string'
    },
    'Polygon': {
        'polygonID': 'string',
        'name': 'string',
        'pop': 'float64',
        'type': 'category',
        'wkt': 'string',
        'link': 'string',
        'notes': 'string'
    },
    'CPHD': {
        'cphdID': 'string',
        'reporterID': 'string',
        'polygonID': 'string',
        'date': 'datetime64[ns]',
        'type': 'category',
        'dateType': 'category',
        'value': 'float64',
        'notes': 'string'
    }
}


UNKNOWN_TOKENS = [
    "nan",
    "na",
    "nd"
    "n.d",
    "none",
    "-",
    "unknown",
    "n/a",
    "n/d"
]

EXCEL_TEST_FILE_PATH = "Data/Ville de Québec 202102.xlsx"

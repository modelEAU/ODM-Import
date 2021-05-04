import pandas as pd 
from wbe_odm.odm_mappers.base_mapper import BaseMapper


default_site_measurement = {
    "siteMeasureID": None,
    "siteID" : None,
    "reporterID": "CityQC",
    "instrumentID": "Pluvio_VilledeQuebec",
    "dateTime" : None,
    "type": "envRnF",
    "aggregation": "single",
    "aggregationDesc": "Cumulative rainfall in one day.",
    "value" : None,
    "unit": "mm",
    "accessToPublic": "No",
    "accessToAllOrgs": "No",
    "accessToPHAC": "No",
    "accessToLocalHA": "No",
    "accessToProvHA": "No",
    "accessToOtherProv": "No",
    "accessToDetails": "No",
    "notes" : ""
}


class VdQRainMapper(BaseMapper):
    def read(self, filepath):
        xl_file = pd.ExcelFile(filepath)
        odm_name = self.conversion_dict["site_measure"]["odm_name"]
        second_sheet_name = xl_file.sheet_names[1]
        xls = pd.read_excel(
            filepath,
            sheet_name=[second_sheet_name],
            header=0,
            skiprows=[1])

        df = xls[second_sheet_name][['Date', 'Pluvio', 'Hauteur totale (mm)']]
        df = df.dropna(subset=['Date'], axis=0)
        df = df.reset_index(drop=True)

        df['siteID'] = pd.Series(map(lambda x : 'QC_wstation_' + str(x), df['Pluvio']))
        del df['Pluvio']

        for k, v in default_site_measurement.items():
            if v is not None :
                df[k] = v

        df['siteMeasureID'] = df['siteID'] +"_"+ df['type'] + "_" \
                            + df["Date"].dt.strftime('%Y-%m-%d')
        df['siteMeasureID'].astype(str)
        df.rename(columns={ 'Hauteur totale (mm)' : 'value', "Date": "dateTime"}, inplace=True)
        df = df[list(default_site_measurement.keys())]
        site_measure = self.type_cast_table(odm_name, df)
        self.site_measure = site_measure
        
    def validates(self):
        return True

if __name__ == '__main__' :
    mapper = VdQRainMapper()
    mapper.read('/mnt/c/Users/medab/Downloads/Pluvio.xlsx')
    #print(mapper.site_measure.head(10))
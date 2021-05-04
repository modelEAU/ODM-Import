import pandas as pd 
from base_mapper import BaseMapper


default_site_measurement = {
    "siteMeasureID": None,
    "siteID" : None,
    "reporterID": " ",
    "instrumentID": "Pluvio_VilledeQuebec",
    "dateTime" : None,
    "type": "envRnF",
    "aggregation": "single",
    "aggregationDesc": "Cumulative rainfall in one day.",
    "value" : None,
    "unit": "mm",
    "accessToPublic": "No",
    "accessToAllOrg": "No",
    "accessToPHAC": "No",
    "accessToLocalHA": "No",
    "accessToProvHA": "No",
    "accessToOtherProv": "No",
    "accessToDetails": "No",
    "notes" : " "
}


class VdQRainMapper(BaseMapper):
    def read(self, filepath):
        sheet_names = ["Pluvio Hiver - Janvier 2021", "Pluvio Hiver brut-Janvier 2021"]
        odm_name = self.conversion_dict["site_measure"]["odm_name"]
        xls = pd.read_excel(
            filepath,
            sheet_name=sheet_names,
            header=0,
            skiprows=[1])

        df = xls["Pluvio Hiver brut-Janvier 2021"][['Date', 'Pluvio', 'Hauteur totale (mm)']]
        df.dropna(subset=['Date'], axis=0)
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
    mapper.site_measure.head(10)
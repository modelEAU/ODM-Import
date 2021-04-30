import pandas as pd
from wbe_odm.odm_mappers import base_mapper


class ExcelTemplateMapper(base_mapper.BaseMapper):
    def __init__(self):
        dico = self.conversion_dict
        dico["sample"]["source_name"] = "Sample"
        dico["ww_measure"]["source_name"] = "WWMeasure"
        dico["site"]["source_name"] = "Site"
        dico["site_measure"]["source_name"] = "SiteMeasure"
        dico["reporter"]["source_name"] = "Reporter"
        dico["lab"]["source_name"] = "Lab"
        dico["assay_method"]["source_name"] = "AssayMethod"
        dico["instrument"]["source_name"] = "Instrument"
        dico["polygon"]["source_name"] = "Polygon"
        dico["cphd"]["source_name"] = "CPHD"
        self.conversion_dict = dico

    def read(
        self,
        filepath,
        sheet_names=None,
            ) -> bool:
        """Reads an ODM-compatible excel file and validates it

        Parameters
        ----------
        filepath : str
            [description]
        sheet_names : [type], optional
            [description], by default None
        """
        import warnings

        if sheet_names is None:
            sheet_names = [
                self.conversion_dict[x]["source_name"]
                for x in self.conversion_dict.keys()]

        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            xls = pd.read_excel(filepath, sheet_name=sheet_names)
        attributes = []
        odm_names = []
        for sheet in sheet_names:
            for attribute, names in self.conversion_dict.items():
                if names["source_name"] == sheet:
                    attributes.append(attribute)
                    odm_names.append(names["odm_name"])

        for attribute, odm_name, sheet in zip(
            attributes,
            odm_names,
            sheet_names,
        ):
            df = xls[sheet]
            # catch breaking change in data model
            if sheet == "WWMeasure" and "assayMethodID" in df.columns:
                df.rename(
                    columns={
                        "assayMethodID": "assayID"
                    },
                    inplace=True
                )
            df = self.type_cast_table(odm_name, df)
            df.drop_duplicates(keep="first", inplace=True)
            setattr(self, attribute, df)
        self.remove_duplicates()
        return

    def validates(self):
        return True

from wbe_odm.odm import Odm, OdmEncoder
from wbe_odm.odm_mappers.excel_template_mapper import ExcelTemplateMapper


TEST_EXCEL_FILE = "tests/test_inputs/Ville de Quebec - All data - v1.1.xlsx"

TEST_DB = "tests/test_data/test_wbe.db"


def test_samples_from_excel():
    # run with example excel data
    filename = TEST_EXCEL_FILE
    excel_mapper = ExcelTemplateMapper()
    excel_mapper.read(filename)
    odm_instance = Odm()
    odm_instance.load_from(excel_mapper)
    geo = odm_instance.get_geoJSON()
    samples = odm_instance.combine_per_sample()
    return geo, samples, odm_instance

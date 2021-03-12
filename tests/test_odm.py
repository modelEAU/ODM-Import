import pytest
from odm_py import odm
from odm_py import constants


@pytest.mark.parametrize(
    "element, expected",
    [
        ('snow', 'water'),
        ('tin', 'solder')
    ]
)
def test_load_from_excel():
    o = odm.Odm()
    o.load_from_excel(constants.EXCEL_TEST_FILE_PATH)
    assert o.data is not None
    assert len(o.data.keys()) == 6

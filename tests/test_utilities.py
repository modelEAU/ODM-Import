import pytest
from odm import constants


def test_types():
    assert len(constants.TYPES.keys()) == 6

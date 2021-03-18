from setuptools import setup, find_packages

setup(
    name="wbe_odm",
    version="0.2.0",
    install_requires=[
        "pandas>=1.2",
        "numpy",
        "sqlalchemy",
        "shapely",
        "geojson_rewind",
    ],
    packages=find_packages(include=[
        "odm",
        "utilities",
        "visualization_helpers",
        "odm_mappers",
        "wbe_tools"
    ]),
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
)

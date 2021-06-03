from setuptools import setup, find_packages

setup(
    name="wbe_odm",
    version="0.5.0",
    install_requires=[
        "pandas>=1.2",
        "unidecode",
        "numpy",
        "sqlalchemy",
        "shapely",
        "geojson_rewind",
        "plotly>4.0",
        "dash>1.0",
        "jupyter",
        "jupyter-dash",
        "geojson",
        "geojson-rewind",
        "geomet",
        "pyyaml",
        "easydict",
        "argparse",
    ],
    packages=find_packages(),
    setup_requires=['pytest-runner', 'flake8'],
    include_package_data=True,
    tests_require=['pytest'],
)

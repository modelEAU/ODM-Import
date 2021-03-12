from setuptools import setup, find_packages

setup(
    name="odm",
    version="0.1.0",
    install_requires=[
        "pandas>=1.2",
        "numpy",
        "sqlalchemy",
        "shapely",
        "geojson_rewind",
        "pygeoif",
        "geodaisy",
    ],
    packages=find_packages(include=[
        "odm",
        "constants",
        "utilities",
        "visualization_helpers"
    ]),
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    package_data={'odm': ['resources/types.json']}
)

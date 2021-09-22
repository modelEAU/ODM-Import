# McGill Lab Sheet Mapper

The csv mapper is used to collect data about any component of the data model contained in data that is in a tabular format in Excel.

## How it works

The Csv Mapper expects two data files:

* A data file that contains data that is constantly updated with new samples and new measurements.
* A static file that defines contants used in the lab file (data about the sites where samples are collected, for example.) The Mapper expects the static file to be in the ODM format.

### The `csv_map` file

The mapper file is made up of rows that each represent a piece of information that needs to be added to ODM tables for each row of the data file.

Each row of the a lab file may contain information about any of the following things:

1. Changes to the Assay Method for SARS detection
1. The names of the reporters performing collection and analysis
1. The sample information
1. The information for each measurement being performed on a given sample

For each of these items, a new row could potentially be generated in the corresponding table of the ODM. Each of these fields are filled in with info of the static file, the lab file, or some constant value that applies to a whole sheet.

For each map row, the corresponding information is present:

* `elementName`: A unique name for a group of mapfile rows that together create a complete row in the ODM.
* `table`: The ODM where new entries will be created.
* `variableName`: The name of the ODM table column where new information will be placed.
* `defaultValue`: Contains a default value for that field.
* `inputSources`: This defines where the mapper object should look for inputs to create the final value that will be placed in the ODM table. The following options are available:
  * `static [ODM table name]`: The info will be found in static data file, in the table whose name is inserted after the word `static`.
  * `lab sheet`: The info will be found in the lab file itself.
  * `[empty]`: The info will be found either in the `defaultValue`field or not at all.
  * `static [ODM table name]+lab sheet`: Required info is in both the static file and the lab sheet.
  
* `labInputs`: This values in this column defines precisely where the mapper object should look for its inputs that don't come from the static file. They are declared in a semi-colon separated list. This list can contain any of the following items:
  * **Excel-style alphabetical column headers**: These items let the mapper object know to use a specific column in the lab sheet as an input. Since the names of the columns of the McGill Lab sheet are not always unique, can get fairly long, the Excel-Style Address is used as an identifier instead of the column name itself.
  * **\_\_Special keywords\_\_**: These keywords let the mapper file know that it should use a constant value from a given source. The accepted keywords are the following:
    * `__default__`: Feeds to the mapper the value in the mapfile's corresping `defaultValue` column.
    * `__labID__`: Feeds to the mapper the ID of the lab maintaing the sheet.
    * `__const__[value]:[Python type for value]`: Feeds to the mapper the value directly following the `__const__` keyword. The type of the value is also specified.
* `processingFunction`: This colun lets the mapper know what function to use to generate the correct value for the field. The mapper will feed all the arguments defined by the `defaultValue`, `inputSources` and `labInputs` columns to that function and assign the output of the function as the value to add to the ODM table for the field defined by `variableName`. The processing functions are defined in the file `mcgill_mapper.py`.

## How to use the mapper

At its core, the csv mapper can be used like any other ODM mapper - by using the `.read()` method.

```python
# Instantiate the mapper
# McGillMapper is a class derived from csv mapper
mapper = McGillMapper()

#Link to the files you want to use.
lab_data = "/Path/to/lab/workbook.xlsx"
static_data = "/Path/to/static/file.xlsx"
mapping = "Data/Lab/McGill/mcgill_map.csv"

# Declare the name of the lab sheet you want to process
lab_worksheet_name = "QC Data Daily Samples (McGill)"

#Declare what lab is rsponsible for that lab sheet
lab_id = "frigon_lab"

#Declare the time range you want to parse the lab data for.    
start_date = "2021-01-01"
end_date = None

# Read the lab file with the help of the supporting files.
mapper.read(
    lab_data,
    static_data,
    mapping,
    lab_worksheet_name,
    lab_id=lab_id,
    startdate,
    enddate
)

# Instantiate the Odm object that will hold the data
odm_object = odm.Odm()

# Load the mapped data into the Odm object.
odm_object.load_from(mapper)
```

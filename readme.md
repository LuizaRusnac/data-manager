# Data Manager

Data Manager is a Python package designed to provide a comprehensive suite of tools specialized in managing and analyzing data. It offers a user-friendly toolkit suitable for individuals of all levels of expertise. This initial release encompasses the following features:

- Effortless reading of datafiles with support for Pandas-compatible extensions.
- Simplified dataframe creation by specifying column names and corresponding values.
- Automatic generation of hexadecimal and numeric identifiers.
- Provision of common statistical analyses for specified columns.
- Histogram plotting functionality.
- Easy visualization of class distribution within the database.
- Data standardization capabilities.
- Feature discretization support.
- Creation of fold columns for conducting stratified k-fold analyses.

# Components
## 'CreateDataPD' Class
This class enables the creation and manipulation of Pandas DataFrames. It supports the following functionalities:
- Reading data from various file formats: CSV, Excel, JSON, Parquet, Feather, and Pickle.
- Generating unique identifiers for rows.
- Adding new data to the DataFrame.
- Displaying the DataFrame.
- Retrieving column names.
- Printing specific columns.

## 'PDNumericAnalysis' Class
This class extends the CreateDataPD class and provides additional functionalities for numerical analysis and preprocessing. It includes the following features:

- Computing statistics for numerical columns.
- Plotting histograms.
- Analyzing data balance.
- Standardizing numerical data.
- Reconstructing data.
- Converting non-numeric columns to numeric values.
- Creating folds for cross-validation.

## Dependencies
This toolkit requires the following dependencies:

- pandas
- numpy
- matplotlib
- scikit-learn

## Installation
To install the dependencies, run:
```python
pip install requirements.txt
```

## Usage
You can use this toolkit in your Python projects by importing the necessary classes and functions. Here's an example of how to use the CreateDataPD class:

```python
from data_analysis_toolkit import CreateDataPD

# Create a DataFrame from data
data = [[1, 'A', 10], [2, 'B', 20], [3, 'C', 30]]
columns = ['ID', 'Category', 'Value']
data_pd = CreateDataPD(data=data, columns=columns)

# Display DataFrame
data_pd.show_dataset()

# Add new data
new_data = [[4, 'D', 40], [5, 'E', 50]]
data_pd.add_data(new_data)

# Display DataFrame with added data
data_pd.show_dataset()
```

# Work in progress..

Stay tuned for upcoming releases, which will incorporate the following enhancements:
- Missing data completion functionality.
- Conversion of dataframes into datasets suitable for machine learning inputs.
- Introduction of PDTextAnalysis for handling features related to text manipulation.
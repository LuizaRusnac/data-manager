import pandas as pd
import numpy as np
import random
import math
from Scripts.plotData import CustomPlot
from sklearn.model_selection import StratifiedKFold

class CreateDataPD:
    def __init__(self, *args, **kwargs):
        self.dataframe = None
        columns = args if args else kwargs.get("columns", None)
        data = kwargs.get("data", None)
        id_type = kwargs.get("id_type", False)
        filename = kwargs.get("filename", None)

        if filename:
            self.read_file(filename)
        else:
            if data is not None:
                self.create_dataframe(data, columns)
            else:
                raise ValueError("Either provide data or columns and data or a valid filename to create DataFrame.")

        if id_type:
            self.complete_id_column(id_type)

    def read_file(self, filename):
        file_extension = filename.split('.')[-1].lower()
        supported_extensions = ['csv', 'xlsx', 'xls', 'json', 'parquet', 'feather', 'pickle']
        
        if file_extension in supported_extensions:
            read_function = getattr(pd, f"read_{file_extension}")
            self.dataframe = read_function(filename)
        else:
            raise ValueError("Unsupported file extension. Supported extensions are: {}".format(', '.join(supported_extensions)))

    def create_dataframe(self, data, columns):
        if data.any():
            self.dataframe = pd.DataFrame(data, columns=columns)
        else:
            raise ValueError("Data is empty.")

    def complete_id_column(self, id_type):
        if not self.check_id_column():
            self.dataframe.insert(0, "id", None)
        self.generate_id(id_type)

    def check_id_column(self, column_name="id"):
        return column_name in self.dataframe.columns

    def generate_id(self, gen_type="hexa"):
        if gen_type == "hexa":
            length_required = max(8, math.ceil(math.log(len(self.dataframe), 16)) + 1)
            self.dataframe.loc[self.dataframe["id"].isna(), "id"] = self.generate_unique_hexa_ids(length_required)
        elif gen_type == "numeric":
            length_required = max(7, math.ceil(math.log10(len(self.dataframe))))
            self.dataframe.loc[self.dataframe["id"].isna(), "id"] = self.generate_unique_numeric_ids(length_required)
        else:
            raise AttributeError("Unknown generation id type")

    def generate_unique_numeric_ids(self, length):
        return np.random.randint(10 ** (length - 1), (10 ** length) - 1, size=len(self.dataframe))

    def generate_unique_hexa_ids(self, length):
        return [hex(random.getrandbits(length * 4))[2:] for _ in range(len(self.dataframe))]

    def show_dataset(self, head=None):
        if head:
            print(self.dataframe.head(head))
        else:
            print(self.dataframe)

    def add_data(self, new_data, regenerate_id=False):
        new_dataframe = pd.DataFrame(new_data, columns=self.dataframe.columns)
        self.dataframe = pd.concat([self.dataframe, new_dataframe], ignore_index=True)
        if regenerate_id:
            self.dataframe["id"] = None
            self.complete_id_column()

    def print_columns_name(self):
        print("Columns:", self.dataframe.columns.tolist())

    def print_columns(self, columns_name):
        print(self.dataframe[columns_name])

class PDNumericAnalysis(CreateDataPD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtypes = self.dataframe.dtypes.reset_index(name='Data_Type')
        self.dictionary = {}

    def column_statistics(self, column_name, **kwargs):
        bins = kwargs.get("bins", 10)
        hist, bins = np.histogram(self.dataframe[column_name], bins=bins)
        stats_dict = {
            'mean': self.dataframe[column_name].mean(),
            'std': self.dataframe[column_name].std(),
            'minim': self.dataframe[column_name].min(),
            'maxim': self.dataframe[column_name].max(),
            'unique_values': self.dataframe[column_name].unique(),
            'histogram': {'hist': hist, 'bins': bins},
            'None_values': {'nr_values': self.dataframe[column_name].isna().sum(),
                            'idxs': self.dataframe[column_name].isna().tolist()}
        }
        return stats_dict

    def plot_histogram(self, column_name, **kwargs):
        plot = CustomPlot()
        plot.histogram_plot(self.dataframe[column_name], **kwargs)
        plot.show_plot()

    def data_balance(self, label_column="labels", **kwargs):
        chart_type = kwargs.get("chart_type", "pie")
        labels_distribution = self.dataframe[label_column].value_counts(sort=False)
        data_nr = len(self.dataframe[label_column])
        labels = self.dataframe[label_column].unique()

        plot = CustomPlot("Class distribution", "", "")
        if chart_type == "bar":
            plot.plot_bar(labels_distribution, label=labels, **kwargs)
        elif chart_type == "pie":
            plot.plot_pie(labels_distribution.values, labels_distribution.index, **kwargs)
        else:
            raise ValueError("Unknown chart type")
        plot.show_plot()

    def column_standardisation(self, column_name, range=[0, 1], overwrite=True):
        minim, maxim = self.dataframe[column_name].min(), self.dataframe[column_name].max()
        if overwrite:
            self.dataframe[column_name] = [(x - minim) * (range[1] - range[0]) / (maxim - minim) + range[0] for x in self.dataframe[column_name]]
        else:
            self.dataframe[f"{column_name}_normalized"] = [(x - minim) * (range[1] - range[0]) / (maxim - minim) + range[0] for x in self.dataframe[column_name]]
        return minim, maxim

    def data_standardisation(self, columns_name, range=[0,1], overwrite=True):
        for column_name in columns_name:
            self.column_standardisation(column_name, range=range, overwrite=overwrite)

    def column_reconstruct(self, column_name, range=[0,1], overwrite=True):
        if overwrite:
            self.dataframe[column_name] = [(x - range[0]) * (self.maxim_columns[column_name] - self.minim_columns[column_name]) / (range[1] - range[0]) + self.minim_columns[column_name] for x in self.dataframe[column_name]]
        else:
            self.dataframe[f"{column_name}_reconstructed"] = [(x - range[0]) * (self.maxim_columns[column_name] - self.minim_columns[column_name]) / (range[1] - range[0]) + self.minim_columns[column_name] for x in self.dataframe[column_name]]

    def data_reconstruction(self, columns_name, range=[0,1], overwrite=True):
        for column_name in columns_name:
            self.column_reconstruct(column_name, range=range, overwrite=overwrite)

    def column_to_numeric_values(self, column_name, overwrite=True):
        unique_values = self.dataframe[column_name].unique()
        self.dictionary[column_name] = {"values": unique_values, "number_values": list(range(len(unique_values)))}
        mapping = {value: index for index, value in enumerate(unique_values)}
        self.dataframe[column_name] = self.dataframe[column_name].map(mapping) if overwrite else self.dataframe[f"{column_name}_numeric"]

    def non_numeric_columns_to_numeric(self, overwrite=True):
        non_numeric_columns = self.find_non_numeric_columns()
        for column in non_numeric_columns:
            self.column_to_numeric_values(column, overwrite=overwrite)

    def find_non_numeric_columns(self):
        return [column for column in self.dataframe.columns if not pd.api.types.is_numeric_dtype(self.dataframe[column]) or self.dataframe[column].dtype == bool]

    def find_numeric_columns(self):
        return [column for column in self.dataframe.columns if pd.api.types.is_numeric_dtype(self.dataframe[column]) and self.dataframe[column].dtype != bool]

    def extract_numeric_data(self):
        return self.dataframe[self.find_numeric_columns()]

    def delete_column(self, columns_name):
        self.dataframe.drop(columns_name, axis=1, inplace=True)

    def create_folds(self, nr_folds=5, label_column="label"):
        skf = StratifiedKFold(n_splits=nr_folds, shuffle=True, random_state=42) 
        self.dataframe["fold"] = -1  
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.dataframe, self.dataframe[label_column])):
            self.dataframe.loc[val_idx, 'fold'] = fold
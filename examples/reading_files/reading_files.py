#DFORGE support reading the following extentions: ['csv', 'xlsx', 'xls', 'json', 'parquet', 'feather', 'pickle']

import dforge as df

json_file = df.PDBuilder(r".\examples\reading_files\json_file.json")
json_file.show_data(5)

xlsx_file = df.PDBuilder(r".\examples\reading_files\xlsx_file.xlsx", index_col=0)
xlsx_file.show_data(5)

parquet_file = df.PDBuilder(r".\examples\reading_files\parquet_file.parquet")
parquet_file.show_data(5)

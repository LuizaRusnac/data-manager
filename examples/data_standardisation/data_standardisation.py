import dforge as df

data = df.PDNumPro(r".\examples\data_standardisation\example.csv")

# Data read from file
data.show_data()

#Non-numeric columns to numeric
data.non_numeric_columns_to_numeric()
data.show_data()

# Data standardisation in range [0,1]
data.data_standardisation()
data.show_data()

# Data reconstruction
data.data_reconstruction()
data.show_data()


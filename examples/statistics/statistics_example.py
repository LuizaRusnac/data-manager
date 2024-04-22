import dforge as df

data = df.PDNumPro(r".\examples\statistics\example.csv")

# Show statistics for column Age
print("\nShow statistics for column Age")
data.show_numeric_columns_statistics('Age')

#Show statistics for columns Age and Height
print("\n\nShow statistics for columns Age and Height")
data.show_numeric_columns_statistics(['Age', 'Height'])

#Show statistics for all numeric columns
print("\n\nShow statistics for all numeric columns")
data.show_numeric_columns_statistics('Numeric')

#Trying to show statistics for non-numeric columns
print("\n\nShow statistics for a numeric column and a non-numeric column")
data.show_numeric_columns_statistics(['Age', 'FAVC'])
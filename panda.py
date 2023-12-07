# basic_pandas_functions.py

import pandas as pd
from matplotlib import pyplot as plt

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'Los Angeles']}
df = pd.DataFrame(data)

#  Display the DataFrame
print("DataFrame:")
print(df)

#  Read Data from CSV File
csv_data = pd.read_csv('example.csv')

#  Display Summary Statistics
summary_stats = df.describe()
print("\nSummary Statistics:")
print(summary_stats)

#  Select a Column
ages = df['Age']
print("\nAges Column:")
print(ages)

#  Filter Data
young_people = df[df['Age'] < 30]
print("\nYoung People:")
print(young_people)

#  Sort Data
sorted_df = df.sort_values(by='Age', ascending=False)
print("\nSorted DataFrame:")
print(sorted_df)



#  Missing Values - Fillna
df_filled = df.fillna(0)
print("\nDataFrame with Missing Values Filled:")
print(df_filled)

#  Drop Columns
df_dropped = df.drop('City', axis=1)
print("\nDataFrame with 'City' Column Dropped:")
print(df_dropped)

#  Apply Function to DataFrame
df['Age_Squared'] = df['Age'].apply(lambda x: x**2)
print("\nDataFrame with Applied Function:")
print(df)

#  Merge DataFrames
df2 = pd.DataFrame({'Name': ['Alice', 'Bob', 'David'],
                    'Salary': [50000, 60000, 70000]})
merged_df = pd.merge(df, df2, on='Name', how='left')
print("\nMerged DataFrame:")
print(merged_df)

#  Pivot Table
pivot_table = df.pivot_table(values='Age', index='City', aggfunc='mean')
print("\nPivot Table:")
print(pivot_table)

#  Plot Data
df.plot(x='Name', y='Age', kind='bar', title='Age Distribution')
plt.show()

#  Save DataFrame to CSV
df.to_csv('output.csv', index=False)



# Project Author: Stephen Mathew
# stephen-mathew.github.io

# Exploratory Data Analysis (EDA) is a crucial step in 
# the data analysis process that involves examining
#  and visualizing datasets to understand their 
# main characteristics. 
# The primary goals of EDA include 
# identifying patterns, 
# spotting anomalies, 
# testing hypotheses, and 
# generating insights 
# before applying more advanced analytical techniques.


### Numpy
import numpy as np

lst = list(range(1000000))
arr = np.arange(1000000, dtype='i4')


# Numpy is faster than Python list
%timeit sum(lst)

%timeit arr.sum()
%timeit np.sum(arr)
%timeit sum(arr)

## Slicing
# Numpy array is more memory efficient
arr = np.arange(24, dtype='i4')
arr2 = arr.reshape((3,8))
arr3 = arr[3::3]
arr[::-3]
np.info(arr), np.info(arr2), np.info(arr3)

## Indexing
arr2
arr2[1]
arr2[1][[1,2]]
arr2[1,[1,2]]
arr2[1][::-1]

arr3
arr3[[1,3,4]]

arr3 % 6 == 0
arr3[arr3 % 6 == 0]
arr3
arr3[arr3 % 6 == 0] = 0
arr3

arr2[1,arr2[1] % 2 == 0]
arr2[1][arr2[1] % 2 == 0]

### Pandas
import pandas as pd
import glob

## Numpy is a dependecy of Pandas
print(pd.show_versions())

# Find all csv files starting with 't20'
csv_files = glob.glob('t20*.csv')
dfs = []
for file in csv_files:
	df = pd.read_csv(file)
	df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y')
	dfs.append(df)

## Multiple assignment / Unpacking
# Store the dataframes
if dfs[0]["match_id"][0] == '202201':
	df2022, df2024 = dfs
else:
	df2024, df2022 = dfs

## Basic dataframe functions
df2022.head()
df2024.head()

df2022.info()
df2022.describe()

df2024.info()

## Pandas Data types and reducing storage (converting to category)
df2022.dtypes

(
    df2022
    .memory_usage(deep=True)
    .sum() / (1024 * 1024)
).round(2).astype(str) + " MB"

(
	df2022
	.select_dtypes(int)
	.describe()
)

(
	df2022
	.venue
	.value_counts()
)

(
	(
	df2022
	.astype({'venue':'category'})
	# .dtypes
	.memory_usage(deep=True)
	.sum() / (1024 * 1024)
	)
	.round(2).astype(str) + " MB"
)

## Chaining
# Months during which the 2022 and 2024 world cups were played
(
	df2022['date']
	.dt
	# .month
	.strftime('%B')
	.unique()
	.tolist()
)
(
	df2024['date']
	.dt
	# .month
	.strftime('%B')
	.unique()
	.tolist()
)

## loc and iloc
(
	df2024
	.set_index('date')
	.loc['2024-06-04 00:00:00']
	# .iloc[850:870:,]
)

## Filtering using loc or query
cols_to_keep = [col for col in df.columns if col != 'match_id']

(
	df2024
	.loc[(df2024['phase']== 'Group B')& (df2024['striker']
			.str.contains('warner', case=False)),cols_to_keep]
)
(
	df2024
	.query("striker.str
		.contains('warner', case=False) and phase == 'Group B'")[cols_to_keep]
)


## Simple group by, pivot and unstack
(
	df2024
	.groupby(['match_id','innings'])['runs_of_bat']
	.agg('sum')
	# .unstack()
)
(
	df2024
	.pivot_table(columns=['match_id','innings'], 
			  values='runs_of_bat', aggfunc='sum')
	# .melt()
	# .stack()
	# .transpose()
)

(
	df2024
	.query("innings in [3,4]")
)

## Sorting
(
	df2024
	.groupby(['batting_team'])['runs_of_bat']
	.agg('sum')
	.sort_values(ascending=False)
)

(
	df2024
	.pivot_table(columns=['batting_team'], values='runs_of_bat', aggfunc='sum')
	.transpose()
	.sort_values(by='runs_of_bat',ascending=False)
)

## Joining 2 dataframes
team_runs_2022 = (
	df2022
	.groupby(['batting_team'])['runs_of_bat']
	.agg('sum')
	.sort_values(ascending=False)
	.reset_index()
)

team_runs_2024 = (
	df2024
	.groupby(['batting_team'])['runs_of_bat']
	.agg('sum')
	.sort_values(ascending=False)
	.reset_index()
)

(
	team_runs_2022
	.merge(team_runs_2024,
		how='outer',
		on='batting_team',
		suffixes=('_2022', '_2024')
		)
	.fillna(0) 					
)

## Group by, Assign, Lambda function, Rename, Concat, Axis, Pipe, Map
# Function to process a DataFrame
def process_df(df):
    return (
        df.rename(columns={'over': 'Over Ball'})
        .assign(over=lambda x: x['Over Ball']
				.apply(lambda y: int(str(y).split('.')[0])+1),
                ball=lambda x: x['Over Ball']
				.apply(lambda y: int(str(y).split('.')[1])))
        .groupby(['match_id','over'])[['runs_of_bat', 'extras']]
        # .agg(['sum', 'mean'])
		.agg('sum')
		.reset_index()
		.groupby(['over'])[['runs_of_bat', 'extras','extras']]
		.agg({
		   'runs_of_bat': ['mean'],
    		'extras': ['mean', 'sum']
		})
    ).map(lambda x: round(x) if isinstance(x, float) else x)

(
	df2024.rename(columns={'over': 'Over Ball'})
        .assign(over=lambda x: x['Over Ball']
				.apply(lambda y: int(str(y).split('.')[0])+1),
                ball=lambda x: x['Over Ball']
				.apply(lambda y: int(str(y).split('.')[1])))
		# .loc[:,['Over Ball','over','ball']]
		# .head()
		# .groupby(['match_id','over'])[['runs_of_bat', 'extras']]
		# .agg('sum')
		# .reset_index()
)
# Process both DataFrames
result_2022 = process_df(df2022)
result_2024 = process_df(df2024)

# Merge results for comparison
comparison_result = pd.concat([result_2022.add_suffix('_2022'), 
							   result_2024.add_suffix('_2024')]
							   , axis=1)

## Apply vs Map
dftest = pd.DataFrame({
    'A': [1.23, 2.1, 3.232],
    'B': [4.2, 5.43, 6.76]
})

(
	dftest
	.map(lambda element: round(element) 
		 if isinstance(element, float) else element)
)
(
	dftest
	.apply(lambda element: round(element) 
		 if isinstance(element, float) else element)
)
(
	dftest
	.apply(lambda row: pd.Series([round(x) if isinstance(x, float) 
							   else x for x in row]), axis=1)
)

## DataFrame and Series
(
	comparison_result
	.reset_index()
	.set_index(('extras_2022', 'sum_2022'))
	.iloc[0:5:,[0,1]]
	# .iloc[0:5:,0] # Dataframe
	# .iloc[0:5:,[0]]. # Series
	.pipe(lambda x: (print(f"Type: {type(x)}"), x)[1]) #prints type
	# .columns
)
(
	comparison_result
	.reset_index()
	.set_index(('extras_2022', 'sum_2022'))
	# .loc[0:6:,[('over',''),('runs_of_bat_2024', 'mean_2024')]]
	.loc[[41,52,32,37],[('over',''),('runs_of_bat_2024', 'mean_2024')]]
	# .loc[[41,52,32,37],:]
	# .pipe(lambda x: (print(f"Type: {type(x)}"), x)[1]) #prints type
)

## Correlation
correlation_matrix = (
    df2024.select_dtypes(include=['number'])
        .corr(method='spearman')
)

import seaborn as sns
import matplotlib.pyplot as plt

## Heatmaps
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            square=True,
            vmin=-1, vmax=1)

plt.title('Spearman Correlation Matrix')
plt.show()


## Conditional formatting styling
team_runs_2022 = (
	df2022
	.groupby(['batting_team'])['runs_of_bat']
	.agg('sum')
	.sort_values(ascending=False)
	.reset_index()
)

team_runs_2024 = (
	df2024
	.groupby(['batting_team'])['runs_of_bat']
	.agg('sum')
	.sort_values(ascending=False)
	.reset_index()
)

# Formatting when comparing cell values with a scalar.
(
	team_runs_2022
	.merge(team_runs_2024,
		how='outer',
		on='batting_team',
		suffixes=('_2022', '_2024')
		)
	.fillna(0)
	.style
	.map(
		lambda x: 'background-color: yellow' if x < 800 else 'background-color: green',
        subset=['runs_of_bat_2022']
	)
	.map(
		lambda x: 'background-color: yellow' if x < 800 else 'background-color: green',
        subset=['runs_of_bat_2024']
	)
)

# Formatting when comparing two columns
merged_total_runs_compare = (
	team_runs_2022
	.merge(team_runs_2024,
		how='outer',
		on='batting_team',
		suffixes=('_2022', '_2024')
		)
	.fillna(0)
)

(
    merged_total_runs_compare.style.apply(
        lambda row: ['background-color: green'] * len(row) 
		if row['runs_of_bat_2022'] < row['runs_of_bat_2024'] 
		else ['background-color: yellow'] * len(row),
        axis=1
    )
)

## Connecting to a database with Pandas
# !pip install psycopg2-binary pandas

import psycopg2
import pandas as pd

# Connection parameters
host = 'localhost'
dbname = 'your_database_name'
user = 'your_username'
password = 'your_password'
port = 5432

# Establish a connection
conn = psycopg2.connect(
    host=host,
    dbname=dbname,
    user=user,
    password=password,
    port=port
)

# Create a cursor object
cur = conn.cursor()

# SQL query to fetch data
sql_query = "SELECT * FROM your_table_name"

# Execute the SQL query
cur.execute(sql_query)

# Fetch all rows from the query
rows = cur.fetchall()

# Get column names
columns = [desc[0] for desc in cur.description]

# Load data into a DataFrame
df = pd.DataFrame(rows, columns=columns)

# Close the cursor and connection
cur.close()
conn.close()

print(df)

## Pandas 2.0
# !pip install pyarrow
# !pip install --upgrade pyarrow
# !pip show pyarrow

# import pandas as pd
# import glob

csv_files_2 = glob.glob('t20*.csv')
dfs_2 = []
for file in csv_files_2:
	df_2 = pd.read_csv(file, dtype_backend='pyarrow')
	df_2['date'] = pd.to_datetime(df_2['date'], format='%b %d, %Y')
	dfs_2.append(df_2)

# Store the dataframes
if dfs_2[0]["match_id"][0] == '202201':
	df2022_, df2024_2 = dfs_2
else:
	df2024_2, df2022_2 = dfs_2

(
	df2022_2
	.memory_usage(deep=True)
	.sum()
)

df2022_2.dtypes

## Pandas with charts
# Cumulative totals
cum_totals_2024_2 = (
	df2024_2[(df2024_2['match_id']==df2024_2['match_id'].max()) & (df2024_2['innings']==2)]
	.rename(columns={'over': 'Over Ball'})
	.assign(over=lambda x: x['Over Ball'].apply(lambda y: int(str(y).split('.')[0])+1),
                ball=lambda x: x['Over Ball'].apply(lambda y: int(str(y).split('.')[1])))
	.assign(total_runs=lambda x: x['runs_of_bat']+x['extras'])			
	.pivot_table(values='total_runs', columns='over',aggfunc='sum')
	.transpose()
	.sort_values(by='over')['total_runs']
	.cumsum()
	.reset_index()
)

import matplotlib.pyplot as plt

# Create line plot using Matplotlib
plt.plot(cum_totals_2024_2['over'], cum_totals_2024_2['total_runs'])

# Add labels and title for clarity
plt.xlabel('Over')
plt.ylabel('Total Runs')
plt.title('Run chart for South Africa')

custom_ticks = [4] 
custom_ticks=[i for i in range(1,len(cum_totals_2024_2)+1) if i %4 ==0]

plt.xticks(custom_ticks)

# Display the plot
plt.show()

### Other packages such as Polars and DuckDB and PySpark

### Pandas profiling
!pip install -U ydata-profiling

from ydata_profiling import ProfileReport
import pandas as pd
import glob

csv_files = glob.glob('t20*.csv')
dfs = []
for file in csv_files:
	df = pd.read_csv(file)
	df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y')
	dfs.append(df)

if dfs[0]["match_id"][0] == '202201':
	df2022, df2024 = dfs
else:
	df2024, df2022 = dfs

# Generate the profile report
profile = ProfileReport(df2024)

# Save the report to an HTML file
profile.to_file("output_report.html")

# Display the report in a Jupyter Notebook (if applicable)
profile.to_notebook_iframe()
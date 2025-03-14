import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

df = pd.read_csv("StudentPerformanceFactors.csv")  # Loading the data in pandas dataframe

print(df.head())  # Inspecting a few rows to see if the data is loaded correctly

print("total columns", df.shape[1])  # Print the total rows and columns in the dataset
print("total rows", df.shape[0])

print(df.isnull().sum())  # Prints the total number of missing values in the dataset

print(df.info())  # Gives the count of not null values in each column with the datatype of the column

df_cleaned = df.dropna()  # Remove missing values
print("Removing missing value", df_cleaned)  # Handling missing values

plt.figure(figsize=(10, 6))  # Creates a figure with the specified size
sns.boxplot(data=df_cleaned)  # Creates a box plot to see the distribution of the data in all columns
plt.xticks(rotation=90)  # Ensures columns do not overlap with each other
plt.show()  # Visualizing the data using boxplot

# Log transformation to handle skewed distribution
df_cleaned.loc[:, 'Exam_Score'] = np.log1p(df_cleaned["Exam_Score"])

# Removing rows where Hours_Studied is greater than 23 (invalid data)
df_cleaned = df_cleaned[df_cleaned["Hours_Studied"] <= 23]

print(df_cleaned.dtypes)
basic_stats = df_cleaned.describe()
print(basic_stats)  # Prints the basic statistics of the dataset

# Correlation matrix calculation (only for numeric columns)
numeric_columns = df_cleaned.select_dtypes(include=['number']).columns
correlation_matrix = df_cleaned[numeric_columns].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()  # Visualizing the correlation matrix to see the correlation between the variables

# Plot Histograms for numeric columns
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)  # Adjusts grid size based on the number of numerical features
    sns.histplot(df_cleaned[col], bins=20, kde=True, color='skyblue')  # KDE = True gives smooth density curve
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("count")
plt.tight_layout()  # Gives a nice layout to the plot
plt.show()

# Plot Bar Charts for Categorical Features
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

# Adjust the grid size based on the number of categorical columns
rows = (len(categorical_columns) // 3) + 1  # Calculate rows needed based on the number of columns

plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(rows, 3, i)  # Dynamically adjust grid size based on number of features
    sns.countplot(data=df_cleaned, x=col, palette="viridis")  # Countplot gives the count of each category
    plt.xticks(rotation=45)  # Rotates labels for better readability
    plt.title(f"Bar Chart of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Scatter plots for combinations of numerical features
plt.figure(figsize=(12, 8))
combinations = list(itertools.combinations(numeric_columns, 2))  # Generate all unique pairs of numerical features

# Adjust grid size based on the number of combinations
rows = (len(combinations) // 4) + 1  # Calculate rows needed based on the number of combinations
cols = 4  # Fixed number of columns (adjust as needed)

for i, (col1, col2) in enumerate(combinations, 1):  # Loop through all feature pairs
    plt.subplot(rows, cols, i)  # Adjust grid size
    sns.scatterplot(data=df_cleaned, x=col1, y=col2, alpha=0.5, color="blue")
    plt.title(f"{col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)

plt.tight_layout()
plt.show()

# Box plots for numerical features
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)  # Adjust grid size
    sns.boxplot(data=df_cleaned, y=col, color="lightcoral")  # Box plot to check for outliers
    plt.title(f"Box Plot of {col}")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

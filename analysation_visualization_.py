import pandas as pd    #importing all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("StudentPerformanceFactors.csv") #loading the data in pandas dataframe

print(df.head())    #inspecting few rows to see if the data is loaded corectly

print("total coloumn",df.shape[1])   #print the total row and coloumn in the dataset
print("total rows", df.shape[0])

print(df.isnull ().sum())   #prints the total number of missing values in the dataset

print(df.info())   #gives the count of not null values in each column with the datatype of the column

df_cleaned = df.dropna()
print("removing missing value" , df_cleaned)  #handling missing value




plt.figure (figsize=(10,6)) #creates a figure with the specified size
sns.boxplot(data= df_cleaned) #creates a box plot to see the distribution of the data in all coloumns Boxplots help detect outliers
plt.xticks(rotation =90)  #ensures coloumn doesnot overlap with each other
plt.show()   #visualizing the data using boxplot to see the distribution of the data

df_cleaned['Exam_Score']= np.log1p(df_cleaned["Exam_Score"])    #log transformation used because it has a skewed distribution
df_cleaned = df_cleaned[df_cleaned["Hours_Studied"] >= 23]    # removing the vlaues of the row if it is more than 23 because that is not possible
#we can handle the data by using mean = average , mode = frequent value , and median = middle value as well

print(df_cleaned.dtypes)
basic_stats = df_cleaned.describe()
print(basic_stats)  #prints the basic statistics of the dataset


correlation_matrix = df_cleaned.corr()
plt.figure(figsize = (6,4))
sns.heatmap(correlation_matrix, annot = True, cmap= ' coolwarm', vmin=1, vmax=1)
plt.title("Correlation Matrix")
plt.show()  #visualizing the correlation matrix to see the correlation between the variables

numeric_columns = df_cleaned.select_dtypes(include=['number']).columns
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)  # Adjusts grid size based on the number of numerical features
    sns.histplot(df_cleaned[col], bins=20, kde=True, color='skyblue')  #kde =True gives smooth density curve
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("count")
plt.tight_layout()  #gives a nice layout to the plot
plt.show()

#Plot Bar Charts for Categorical Feature
plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_columns, 1):  #enumerate  assigns an index (i) starting from 1 and creates subplot dinamically
    plt.subplot(3, 3, i)  # Adjusts grid size based on the number of categorical features
    sns.countplot(data=df_cleaned, x=col, palette="viridis")  #countplot gives the count of each category
    plt.xticks(rotation=45)  # Rotates labels for better readability
    plt.title(f"Bar Chart of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

import itertools

# Selecting numerical features for scatter plots
numerical_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(12, 8))
combinations = list(itertools.combinations(numerical_columns, 2))  # Generate all unique pairs of numerical features

for i, (col1, col2) in enumerate(combinations, 1):  #This line loops through all feature pairs and assigns values to col1 and col2.
    plt.subplot(4, 4, i)  # Adjust the grid size if needed
    sns.scatterplot(data=df_cleaned, x=col1, y=col2, alpha=0.5, color="blue")
    plt.title(f"{col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)  # Adjust grid size
    sns.boxplot(data=df_cleaned, y=col, color="lightcoral")  # Box plot to check for outliers
    plt.title(f"Box Plot of {col}")
    plt.ylabel(col)

plt.tight_layout()
plt.show()



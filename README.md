import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Now load Dataset
df = pd.read_csv("Titanic_test.csv")
#View Dataset
print("First 5 Rows")
print(df.head())

#Dataset information
print("\nDataset information")
print(df.info())

#Now cheaking missing values
print("\nmissing value before cleaning")

#fill Age row missing  value with  median
df['Age']=df['Age'].fillna(df['Age'].median())
#Now  fill the  fare  with  median
df['Fare']=df['Fare'].fillna(df['Fare'].median())

#Remove Dulicate  Rows
df.drop_duplicates(inplace=True)
#checking  missing  value  again
print(df.isnull().sum())
# 8. Dataset Shape

print("\nDataset Shape:")
print(df.shape)
df.to_csv("Titanic_Cleaned.csv", index=False)
print("\nCleaning Completed Successfully!")
print("Cleaned file saved as Titanic_Cleaned.csv")

print(df.describe())
#Now create a  chart for Gender Distribution
plt.figure(figsize=(6,4))

sns.countplot(x='Sex', data=df)

plt.title("Gender Distribution")
plt.show()

#Now create chart for passenger Distribution

plt.figure(figsize=(6,4))

sns.countplot(x = 'Pclass' , data = df)
plt.title("Passenger Distribution")
plt.show()

#Now create  a  chart  for Fare distribution
plt.figure(figsize=(6,4))
plt.hist(df['Fare'] , bins = 20 ,color = 'green')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

#Now create a  chart for Passenger Class by Gender
plt.figure(figsize=(7,5))

sns.countplot(x='Pclass', hue='Sex', data=df)

plt.title("Passenger Class by Gender")

plt.show()

#show  correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()














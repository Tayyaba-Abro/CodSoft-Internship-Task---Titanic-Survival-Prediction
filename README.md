# CodSoft Internship Task - Titanic Survival Prediction

## Introduction: 
In this repository, you will find a comprehensive analysis of the famous Titanic dataset and a machine learning model that predicts passenger survival. This project aims to explore key factors affecting passenger survival, such as gender, age, passenger class, and embarkation location.

## Quick Links:
Dataset: [Titanic Survival Prediciton Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)

## Understanding Dataset 
1. PassengerID: A unique identifier for each passenger in the dataset. It serves as a primary key for individual records.  
2. Survived: Indicates whether a passenger survived (1) or did not survive (0) the Titanic disaster. This is the target variable for predictive modeling.  
3. Pclass: Represents the passenger class, with three possible values: 1st class (1), 2nd class (2), and 3rd class (3). It reflects the socio-economic status of passengers. 4. Name: The full name of the passenger. It includes both the passenger's title and their name.   
5. Sex: Specifies the gender of the passenger, either male or female.  
6. Age: The age of the passenger in years. It represents the passenger's age at the time of boarding the Titanic.  
7. SibSp: Indicates the number of siblings or spouses that a passenger had aboard the Titanic.  
8. Parch: Represents the number of parents or children that a passenger had aboard the Titanic.   
9. Ticket: The ticket number associated with the passenger's ticket. It is an alphanumeric identifier.  
10. Fare: The fare or price paid by the passenger for their ticket.  

These columns details provide a clear understanding of the dataset's features, which is essential for data analysis, feature engineering, and building predictive models. Each column has a specific role and type of data, and some may be more relevant than others for predicting passenger survival or for other analysis tasks related to the Titanic dataset.

## Task:
The following procedure was followed to conplete titanic survival prediction task:

### 1. Importing Libraries:
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
### 2. Importing Dataset: 
```python
# create a pandas DataFrame to read csv
df = pd.read_csv(r"C:\Users\Asad Ali\Desktop\Courses\Internship\CodSoft\Task 1 - Titanic Survival\titanic.csv")
# show first 5 rows of the DataFrame
df.head()
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/fe8c358c-2082-47f1-b3e2-fdeaa4c3452f)

### 3. Understanding Dataframe:
```python
# check the records and attributes of the DataFrame (rows and columns)
df.shape
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/fbbae38a-7569-4c14-a28b-8e64eedba4eb)

```python
# display basic statistics of the DataFrame
df.describe()
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/e2341610-f09c-4e09-8bbb-212c754a4c18)

```python
# check if there are any null values in the DataFrame
df.isnull().sum()
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/648daa1a-26d8-45f2-9db4-1a1ad196feff)

```python
# check datatypes of all the columns in the DataFrame
df.dtypes
```
![data types](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/fbb33b00-b62d-4499-a6bb-d7ac6c2566d0)

### 4. Preprocessing 
```python
# take mean of Age column and fill the value for all the missing values in that column
df['Age'] = df['Age'].fillna(df['Age'].mean())
# similarly take mean of fare column and fill the missing values
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Fare']
df['Age']
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/b7767e50-8d75-4c24-b9af-7d73b5cef6cd)

```python
# display all the unique values in the column Embarked
Embarked = df['Embarked'].unique()
for Embarkeds in Embarked:
    print("->",Embarkeds)
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/f34073f5-6c7c-4f38-97bc-a8fa1e1166a1)

### 5. Exploratory Data analysis (EDA)
We have performed Exploratory Data analysis (EDA) based on following parameters:

#### i. Calculate Survival Rates for each Sex
```python
survival_rates = df.groupby('Sex')['Survived'].mean()

# Define custom colors for pie chart
colors = ['#ff9999', '#66b3ff']

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(survival_rates, labels=None, autopct='%1.1f%%', startangle=90, colors= colors)
plt.title('Survival Rate by Sex')

# Create a legend
plt.legend(survival_rates.index, title='Sex', loc='upper right', bbox_to_anchor=(1, 1))

# Display the chart
plt.show()
```
![survivalby sec](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/a2f55c2e-5675-4480-88c3-4289a13d3ca2)

#### ii. Calculate the Percentage of Passengers for each Embarkation Category
```python
# Calculate the percentage of passengers for each embarkation category
embarked_percentage = df['Embarked'].value_counts(normalize=True) * 100

# Create a bar chart for embarked percentage
plt.figure(figsize=(8, 5))
bars = plt.bar(embarked_percentage.index, embarked_percentage.values, color=['#ff9999', '#66b3ff', '#99ff99'])
plt.xlabel('Embarkation Category')
plt.ylabel('Percentage')
plt.title('Percentage of Passengers by Embarkation Category')
plt.ylim(0, 100)
plt.bar_label(bars, labels=[f'{percentage:.2f}%' for percentage in embarked_percentage], fontsize=12)

# Display the chart
plt.show()
```
![population by embarked](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/c681dc09-8970-42df-abc8-ff0fe107b50f)

#### iii. Calculate Survival Rates for each Embarkation Category
```python
# Calculate survival rates for each embarkation category
survival_rates = df.groupby('Embarked')['Survived'].mean()

# Define custom colors
colors = ['#ff9999', '#66b3ff', '#99ff99']

# Create a pie chart for survival rate
plt.figure(figsize=(6, 6))
plt.pie(survival_rates, labels=survival_rates.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Survival Rate by Embarkation Category')

# Display the chart
plt.show()
```
![percentage of embarked by surviva](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/22e0660e-6bcc-4548-a2c3-c72b061cb368)

#### iv. Calculate the Percentage of Passengers for each Passenger Class
```python
# Calculate the percentage of passengers for each passenger class
pclass_percentage = df['Pclass'].value_counts(normalize=True) * 100

# Create a bar chart for Pclass percentage
plt.figure(figsize=(8, 5))
bars = plt.bar(pclass_percentage.index, pclass_percentage.values, color=['#ff9999', '#66b3ff', '#99ff99'])
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Percentage')
plt.title('Percentage of Passengers by Passenger Class')
plt.xticks(pclass_percentage.index)
plt.ylim(0, 100)

plt.bar_label(bars, labels=[f'{percentage:.2f}%' for percentage in pclass_percentage], fontsize=12)
# Display the chart
plt.show()
```
![ratio of percentage of class](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/bd1ffb02-219c-444b-9ac1-f681713a451e)

#### v. Calculate Survival Rates for each Passenger Class
```python
# Calculate survival rates for each passenger class
survival_rates = df.groupby('Pclass')['Survived'].mean()

# Define custom colors
colors = ['#ff9999', '#66b3ff', '#99ff99']

# Create a pie chart for survival rate by Pclass
plt.figure(figsize=(6, 6))
plt.pie(survival_rates, labels=survival_rates.index.map({1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}), autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Survival Rate by Passenger Class (Pclass)')

# Display the chart
plt.show()
```

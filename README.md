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
![df head](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/fe8c358c-2082-47f1-b3e2-fdeaa4c3452f)

### 3. Understanding Dataset:
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

```python
# take mean of Age column and fill the value for all the missing values in that column
df['Age'] = df['Age'].fillna(df['Age'].mean())
# similarly take mean of fare column and fill the missing values
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Fare']
df['Age']
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/b7767e50-8d75-4c24-b9af-7d73b5cef6cd)



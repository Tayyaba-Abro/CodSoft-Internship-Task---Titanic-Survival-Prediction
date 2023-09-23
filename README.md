# CodSoft Internship Task - Titanic Survival Prediction

![Slide1](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/81e781c7-d721-44c0-b081-d92b5cc01fbb)

## Introduction: 
In this repository, you will find a comprehensive analysis of the famous Titanic dataset and a machine learning model that predicts passenger survival. This project aims to explore key factors affecting passenger survival, such as gender, age, passenger class, and embarkation location.

## Quick Links:
Dataset: [Titanic Survival Prediction Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)

## Understanding Dataset 
1. PassengerID: A unique identifier for each passenger in the dataset. It serves as a primary key for individual records.  
2. Survived: Indicates whether a passenger survived (1) or did not survive (0) the Titanic disaster. This is the target variable for predictive modeling.  
3. Pclass: Represents the passenger class, with three possible values: 1st class (1), 2nd class (2), and 3rd class (3). It reflects the socio-economic status of passengers.   
4. Name: The full name of the passenger. It includes both the passenger's title and their name.   
5. Sex: Specifies the gender of the passenger, either male or female.  
6. Age: The age of the passenger in years. It represents the passenger's age at the time of boarding the Titanic.  
7. SibSp: Indicates the number of siblings or spouses that a passenger had aboard the Titanic.  
8. Parch: Represents the number of parents or children that a passenger had aboard the Titanic.   
9. Ticket: The ticket number associated with the passenger's ticket. It is an alphanumeric identifier.  
10. Fare: The fare or price paid by the passenger for their ticket.  

These columns details provide a clear understanding of the dataset's features, which is essential for data analysis, feature engineering, and building predictive models. Each column has a specific role and type of data, and some may be more relevant than others for predicting passenger survival or for other analysis tasks related to the Titanic dataset.

## Tasks
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

### 4. Preprocessing: 
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

### 5. Exploratory Data analysis (EDA):
We have performed Exploratory Data analysis (EDA) based on following parts:

#### Part 1: Passenger Demographics and Survival

##### i. Calculate Survival Rates for each Sex
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
<img src="https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/a2f55c2e-5675-4480-88c3-4289a13d3ca2" width="500" height="400">

##### ii. Calculate the Percentage of Passengers for each Embarkation Category
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
<img src="https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/c681dc09-8970-42df-abc8-ff0fe107b50f" width="400" height="500">

##### iii. Calculate Survival Rates for each Embarkation Category
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
<img src="https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/22e0660e-6bcc-4548-a2c3-c72b061cb368" width="400" height="500">

##### iv. Calculate the Percentage of Passengers for each Passenger Class
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
<img src="https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/66c9bd38-063e-4428-b85f-19c02804537a" width="400" height="400">

##### v. Calculate Survival Rates for each Passenger Class
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
<img src="https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/5d4a4144-95c7-4fa3-9327-a4ef01ef8be2" width="400" height="500">

#### Part 2: Data Transformation
In addition to the above procedure, we will map values in the 'Embarked' and 'Sex' columns to integers and change the data types of 'Age' and 'Fare' columns to integers.
```python
# map values from Embarked and Sex columns to integer values and change the datatype
df['Embarked'] = df['Embarked'].map( {'Q': 0,'S':1,'C':2}).astype(int)
df['Sex'] = df['Sex'].map( {'female': 1,'male':0}).astype(int)

# change the datatypes of columns Age and Fare from float to integer
df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(int)

# check the datatypes of all the columns once again to see the changes
df.dtypes
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/0f32caba-77a5-4a99-a843-1881ab9ae775)

```python
# create a copy of df DataFrame without columns PassengerId, Name, Cabin, and Ticket
df.drop(['PassengerId','Name','Cabin','Ticket'], axis =1, inplace=True)
# show first 5 records of the DataFrame
df.head()
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/f2761366-9866-45f0-8c3f-854fc52d10d4)

#### Part 3: Data Filtering and Visualization

Now, we will filter data for survivors and non-survivors
```python
survivors = df[df['Survived'] == 1]
non_survivors = df[df['Survived'] == 0]

# Create bins with a difference of 10
bins = range(0, 91, 10)  # Bins from 0 to 90 with a step of 10

# Create a histogram for the age distribution of survivors
plt.figure(figsize=(10, 6))
plt.hist(non_survivors['Age'].dropna(), bins=bins, color='#ff9999', label='Non-Survivors')
plt.hist(survivors['Age'].dropna(), bins=bins, color='#66b3ff', label='Survivors')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Survivors and Non-Survivors')
plt.legend()
plt.grid(True)

# Display the histogram
plt.show()
```
<img src="https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/68c97a76-85b0-4382-9db9-190e5ca1f44b" width="400" height="500">

### 6. Data Modeling and Training the Model
```python
train = df.drop(['Survived'], axis=1)
test = df.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 1)
```

### 7. Logistic Regression Model
```python
# Logistic regression accuracy score
LR = LogisticRegression(solver='liblinear', max_iter=200)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic regression accuracy: {:.2f}%'.format(LRAcc*100))

# Logistic Regression Plot
LRAcc = 0.9286 

# Create a bar graph
plt.bar(['Logistic Regression'], [LRAcc])
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1 for accuracy percentage

# Add labels
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

# Display the accuracy as text on top of the bar
plt.text('Logistic Regression', LRAcc + 0.02, f'{LRAcc*100:.2f}%', ha='center')

# Show the graph
plt.show()
```
![Logistic Regression](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Titanic-Survival-Prediction/assets/47588244/99652569-fb60-45b9-89e9-c29148048483)

## Conclusion:
In the course of this Titanic survival prediction project, we embarked on a data-driven journey to uncover valuable insights hidden within the Titanic dataset. By following a systematic workflow, we gained a deeper understanding of the factors influencing survival rates and made several key predictions.

-- **Survival by Sex:** Our analysis revealed that women were overwhelmingly preferred for survival. Remarkably, no adult males survived the tragic event.

-- **Age Distribution of Survival & Non-Survivals:** We observed a preference for survival among individuals in the lower to middle age range. Those above 40 were least prioritized for survival

-- **Embarkation Insights:** Passengers who embarked from Queenstown, despite being fewer in number, experienced the highest survival rate. In contrast, Southampton, with a larger passenger population, had the lowest survival rate

-- **Survival by Passenger Class:** First-class passengers had the highest survival rate, emphasizing the privilege associated with higher class. Second-class passengers, on the other hand, faced the lowest survival rate among the other classes.

Throughout this journey, we employed data preprocessing, exploratory data analysis, and logistic regression modeling to derive these insights. While our analysis provides valuable historical context, it's essential to remember that these findings are based on data from a specific event in history and may not generalize to all scenarios. 

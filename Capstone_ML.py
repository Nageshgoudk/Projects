
"""#Machine Learning (Capstone Project)"""

#Firstly most common symptoms are Fever,Cough and Headche for Positive cases these symptoms people must take precautions of covid-19. Symptoms facing like Sore throat,Shortness of breath also have to take corona test.
#Medical staff need to focus on people with Positive cases having breath issues very high.
#Positive people with Fever,Cough and Headche are no need rush on hospitals they just need to follow Covid-19 rules and consume high immunity food.
#Medical team have to focus to make people understanding what covid rules have to follow when they get corona test positive by using socialmedia platform.
#By this medical team have less burden.

import pandas as pd

df=pd.read_csv('/content/corona_tested_006.csv')

for i in df:
  print(i, ':', df[i].unique())

df['Cough_symptoms']=df['Cough_symptoms'].apply(lambda x: x.lower() if isinstance(x,str) else x)

df['Cough_symptoms']=df['Cough_symptoms'].apply(lambda x: 'false' if x==False or x=='none' else x)
df['Cough_symptoms']=df['Cough_symptoms'].apply(lambda x: 'true' if x==True else x)

df['Fever']=df['Fever'].apply(lambda x: x.lower() if isinstance(x,str) else x)

df['Fever'].mode()[0]

df['Fever']=df['Fever'].apply(lambda x: 'false' if x==False or x=='none' else x)
df['Fever']=df['Fever'].apply(lambda x: 'true' if x==True else x)

df['Sore_throat']=df['Sore_throat'].apply(lambda x: x.lower() if isinstance(x,str) else x)

df['Sore_throat']=df['Sore_throat'].apply(lambda x: 'false' if x==False or x=='none' else x)
df['Sore_throat']=df['Sore_throat'].apply(lambda x: 'true' if x==True else x)

df['Shortness_of_breath']=df['Shortness_of_breath'].apply(lambda x: x.lower() if isinstance(x,str) else x)

df['Shortness_of_breath']=df['Shortness_of_breath'].apply(lambda x: 'false' if x==False or x=='none' else x)
df['Shortness_of_breath']=df['Shortness_of_breath'].apply(lambda x: 'true' if x==True else x)

df['Headache']=df['Headache'].apply(lambda x: x.lower() if isinstance(x,str) else x)

df['Headache']=df['Headache'].apply(lambda x: 'false' if x==False or x=='none' else x)
df['Headache']=df['Headache'].apply(lambda x: 'true' if x==True else x)

df['Corona'].mode()

df['Corona']=df['Corona'].apply(lambda x: 'Negative' if x=='negative' or x=='other' else x)

df.isnull().sum()

df['Sex'].mode()

df['Sex']=df['Sex'].apply(lambda x: 'female' if x=='None' else x)

df['Age_60_above']=df['Age_60_above'].apply(lambda x: 'No' if x=='None' else x)

import matplotlib.pyplot as plt

import plotly.express as px
import seaborn as sns

px.imshow((pd.crosstab(df['Corona'],df['Sex'])))

import sklearn

new_df=pd.get_dummies(df['Sex'],drop_first=True)

new_df=pd.get_dummies(df['Sex'],drop_first=False)

new_df

df

df2=pd.concat([df,new_df],axis=1)

df2

import plotly.express as px

df2['year']=pd.to_datetime(df2['Test_date']).dt.year

df2['Corona_score']=pd.get_dummies(df['Corona'],drop_first=True)

df2['Cough_score']=pd.get_dummies(df['Cough_symptoms'],drop_first=True)

df2['Fever_score']=pd.get_dummies(df['Fever'],drop_first=True)

df2['breathShortness_score']=pd.get_dummies(df['Shortness_of_breath'],drop_first=True)

df2['Headache_score']=pd.get_dummies(df['Headache'],drop_first=True)

df2['Sore_throat_score']=pd.get_dummies(df['Sore_throat'],drop_first=True)

df2

"""**accuracy testing**"""

X = df2[['Cough_score', 'Fever_score', 'breathShortness_score','Headache_score','Sore_throat_score']]
y = df2['Corona_score']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=True)

#modelling

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

# Here accuracy is 95%

print("Accuracy Score:",accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#Sql Questions

import duckdb
conn=duckdb.connect()

conn.register('df2',df2)

"""1.Find the number of corona patients who faced shortness of breath"""

conn.execute("select count(Corona_score) as corona_patients from df2 where Shortness_of_breath='true'").fetchdf()

"""2.Find the number of negative corona patients who have fever and sore_throat."""

conn.execute("select count(Corona) as Negative_cases from df2 where Corona='Negative' and Ind_ID in(select Ind_ID from df2 where Shortness_of_breath='true'and Sore_throat='true')").fetchdf()

"""3.Group the data by month and rank the number of positive cases."""

import datetime

conn.execute("select MONTH(Test_date), count(*) as Positive_cases from df2 where Corona='positive' group by MONTH(Test_date) ORDER BY Positive_cases DESC").fetchdf()

"""4.Find the female negative corona patients who faced cough and headache."""

conn.execute("select Ind_ID as Female_ID from df2 where female=1 and Corona='Negative' and Ind_ID in(select Ind_ID from df2 where Headache='true'and Cough_symptoms='true')").fetchdf()

"""5.How many elderly corona patients have faced breathing problems?"""

conn.execute("select count(Corona) as Elder_cases from df2 where Age_60_above = 'Yes' and Corona='positive' and Ind_ID in(select Ind_ID from df2 where Shortness_of_breath='true')").fetchdf()

"""6.Which three symptoms were more common among COVID positive patients?

Here Headache, Fever, Cough are more common amoung positive cases.
"""

conn.execute("select sum(Sore_throat_score), sum(Headache_score), sum(breathShortness_score), sum(Fever_score), sum(Cough_score) from df2 where Corona='positive'").fetchdf()

"""7.Which symptom was less common among COVID negative people?

Here less common symptom is Headache
"""

conn.execute("select sum(Sore_throat_score), sum(Headache_score), sum(breathShortness_score), sum(Fever_score), sum(Cough_score) from df2 where Corona='Negative'").fetchdf()

"""8.What are the most common symptoms among COVID positive males whose known contact was abroad?

Here most common symptoms are Fever,Cough.
"""

conn.execute("select sum(Sore_throat_score), sum(Headache_score), sum(breathShortness_score), sum(Fever_score), sum(Cough_score) from df2 where Corona='positive' and Known_contact='Abroad' and Sex='male'").fetchdf()
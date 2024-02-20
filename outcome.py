#Column defintions:
# id - Unique ID for each home sold
#date - Date of the home sale
#price - Price of each home sold
#bedrooms - Number of bedrooms
#bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
#sqft_living - Square footage of the apartments interior living space
#sqft_lot - Square footage of the land space
#floors - Number of floors
#waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
#view - An index from 0 to 4 of how good the view of the property was 0 = No view, 1 = Fair 2 = Average, 3 = Good, 4 = Excellent
#condition - An index from 1 to 5 on the condition of the apartment,1 = Poor- Worn out, 2 = Fair- Badly worn, 3 = Average, 4 = Good, 5= Very Good
#grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
#sqft_above - The square footage of the interior housing space that is above ground level
#sqft_basement - The square footage of the interior housing space that is below ground level
#yr_built - The year the house was initially built
#yr_renovated - The year of the house’s last renovation
#zipcode - What zipcode area the house is in
#lat - Lattitude
#long - Longitude
#sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
#sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title('Housing Data:')
st.write("This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.")

with st.expander("Column defintions:"):
    st.write("The columns in the dataset are as follows:")
    st.write("id - Unique ID for each home sold")
    st.write("date - Date of the home sale")
    st.write("price - Price of each home sold")
    st.write("bedrooms - Number of bedrooms")
    st.write("bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower")
    st.write("sqft_living - Square footage of the apartments interior living space")
    st.write("sqft_lot - Square footage of the land space")
    st.write("floors - Number of floors")
    st.write("waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not")
    st.write("view - An index from 0 to 4 of how good the view of the property was 0 = No view, 1 = Fair 2 = Average, 3 = Good, 4 = Excellent")
    st.write("condition - An index from 1 to 5 on the condition of the apartment,1 = Poor- Worn out, 2 = Fair- Badly worn, 3 = Average, 4 = Good, 5= Very Good")
    st.write("grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.")
    st.write("sqft_above - The square footage of the interior housing space that is above ground level")
    st.write("sqft_basement - The square footage of the interior housing space that is below ground level")
    st.write("yr_built - The year the house was initially built")
    st.write("yr_renovated - The year of the house’s last renovation")
    st.write("zipcode - What zipcode area the house is in")
    st.write("lat - Lattitude")
    st.write("long - Longitude")
    st.write("sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors")
    st.write("sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors")
# st.write("The goal of this notebook is to predict the price of a house given its features.")



df=pd.read_csv("kc_house_data_1.csv")

df["date"]= pd.to_datetime(df.date)#changing the obeject data type into the datetime


st.write("our dataset has {num_columns} columns and {num_rows} rows".format(num_columns = df.shape[1]
                                                                        ,num_rows = df.shape[0]))
if st.sidebar.checkbox("Show raw data", False):
    st.subheader('Raw data')
    st.write(df)
if st.sidebar.checkbox("Show summary", False):
    st.subheader('Summary')
    st.write(df.describe().T)


st.subheader('Data Cleaning')
st.write('The dataset has some null values.')
st.write('We will fill the null values with the mean of the column.')
st.write('We will also create a new column called house_age which is the difference between the year the house was sold and the year it was built.')
st.write('We will also remove the rows where the house_age is negative as this is noise.')
if st.sidebar.checkbox("Show null values", False):
    st.subheader('Null values')
    st.write(df.isnull().sum())

df["price"]=df["price"].fillna(df["price"].mean())
df["yr_built"]=df["yr_built"].fillna(df["yr_built"].mean())

age_of_house = [df['date'][index].year - df['yr_built'][index] for index in range(df.shape[0])]
df["house_age"] = age_of_house
df.drop(df[df.house_age < 0].index , inplace =True)
df.reset_index(inplace = True , drop =True)



st.subheader('Data Visualization')
st.write('Let us visualize the data to get a better understanding of it.')
st.write('We will use seaborn and matplotlib for this.')
st.write('We will plot the price of the house against the date it was sold.')
st.write('We will also plot the price of the house against the number of bedrooms and bathrooms.')
st.write('We will also plot the price of the house against the number of floors.')
st.write('We will also plot the price of the house against the grade of the house.')
st.write('We will also plot the price of the house against the square footage of the living space.')




# Plot the date and price column
fig = plt.figure(figsize=(8,6))
plt.scatter(df["date"], df["price"])
plt.title("Price vs. Date", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price")
st.write(fig)

# check the percentile and median base distribution(For visulaising the outliers)
fig= plt.figure(figsize=(20,12))

ax=fig.add_subplot(2,2,1)
print(sns.boxplot(data=df, x=df["bedrooms"], y=df["price"], hue=None, color='green', ax=ax))
ax.set_title("Price vs bedrooms ")

ax=fig.add_subplot(2,2,2)
sns.boxplot(data=df, x=df["floors"], y=df["price"], hue=None, color='brown', ax=ax)
ax.set_title("Price vs floors")

ax=fig.add_subplot(2,2,3)
sns.boxplot(data=df, x=df["bathrooms"], y=df["price"], hue=None, color='blue', ax=ax)
ax.set_title("Price vs bathrooms")


ax=fig.add_subplot(2,2,4)
sns.boxplot(data=df, x=df["grade"], y=df["price"], hue=None, color='yellow', ax=ax)
ax.set_title("Price vs grade")

st.write(fig)

# Plot the price and sqft_living column
fig =plt.figure(figsize=(12,10))
plt.scatter(df["sqft_living"], df["price"])
plt.title("Price vs. Sqft_living", fontsize=16)
plt.xlabel("Sqft_living")
plt.ylabel("Price")
st.write(fig)

corelation=df.corr()
if st.sidebar.checkbox("Show corelation", False):
    st.subheader('Corelation')
    st.write(corelation)
    fig=plt.figure(figsize=(12,10))
    sns.heatmap(corelation, annot=True, fmt='.2f')
    st.write(fig)

# Machine Learning Part
st.subheader('Machine Learning')
st.write('We will use the following features to predict the price of the house:')
st.write('bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, house_age')

st.write('We will use the following algorithms to predict the price of the house:')
st.write('Polynomial Regression (Degree = 2)')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


x = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'zipcode', 'house_age']].values #features
y = df['price'].values #target variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=42)#splitting the data into train and test

model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


fig = plt.figure(figsize=(12,10))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
st.write(fig)

st.write('We will use the following metrics to evaluate the performance of the model:')
st.write("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
st.write("R^2 score %.2f"% r2_score(y_test, y_pred))   #R^2 score
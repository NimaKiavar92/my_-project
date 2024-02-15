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
df=pd.read_csv("kc_house_data_1.csv")



print(df.head(10))
print(df.tail(10))
print(df.columns)
print(df.dtypes)
df["date"]= pd.to_datetime(df.date)#changing the obeject data type into the datetime
print("our dataset has {num_columns} columns and {num_rows} rows".format(num_columns = df.shape[1]
                                                                        ,num_rows = df.shape[0]))
print(df.nunique())
print(df.info())
print(df.isnull().sum())


df["date"]= pd.to_datetime(df.date)#changing the obeject data type into the datetime
df["price"]=df["price"].fillna(df["price"].mean())
df["yr_built"]=df["yr_built"].fillna(df["yr_built"].mean())
print(df.isnull().sum())
print(df.describe().T)
df[["price","yr_built"]].groupby(["price"]).mean()
df[["price","bathrooms"]].groupby(["price"]).mean()
df[["price","zipcode"]].groupby(["price"]).mean()
age_of_house = [df['date'][index].year - df['yr_built'][index] for index in range(df.shape[0])]
df["house_age"] = age_of_house
print(df['house_age'].agg({'min','max','mean'}))
print(df[df.house_age < 0])#maybe this data are noise
df.drop(df[df.house_age < 0].index , inplace =True)
df.reset_index(inplace = True , drop =True)
print(df)

# Plot the date and price column
fig = plt.figure(figsize=(8,6))
plt.scatter(df["date"], df["price"])
plt.title("Price vs. Date", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

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

plt.show()

# Plot the price and sqft_living column
fig =plt.figure(figsize=(12,10))
plt.scatter(df["sqft_living"], df["price"])
plt.title("Price vs. Sqft_living", fontsize=16)
plt.xlabel("Sqft_living")
plt.ylabel("Price")
plt.show()
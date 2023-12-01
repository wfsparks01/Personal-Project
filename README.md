# Personal-Project
Predicting Car Prices 
# Make                Car Make
# Model               Car Model
# Year                Car Year (Marketing)
# Engine Fuel Type    Engine Fuel Type
# Engine HP           Engine Horse Power (HP)
# Engine Cylinders    Engine Cylinders
# Transmission Type   Transmission Type
# Driven_Wheels       Driven Wheels
# Number of Doors     Number of Doors
# Market Category     Market Category
# Vehicle Size        Size of Vehicle
# Vehicle Style       Type of Vehicle
# highway MPG         Highway MPG
# city mpg            City MPG
# Popularity          Popularity (Twitter)
# MSRP                Manufacturer Suggested Retail Price

import statistics
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
df.head(5)

print("The dataframe contains",df.shape[0],"rows and",df.shape[1],"columns\n")
print("The labels are",[df.columns[i] for i in range(df.shape[1])], "\n")
df.info()

index = df.groupby(['Year']).mean(numeric_only=True)['MSRP'].index.tolist()
mean_prices = df.groupby(['Year']).mean(numeric_only=True)['MSRP'].values.tolist()
std_prices = df.groupby(['Year']).std(numeric_only=True)['MSRP'].values.tolist()

# Data that will be used
price_per_year = pd.DataFrame(np.column_stack((mean_prices, std_prices)), columns=['Mean', 'Std'], index=index)

# Plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=index, y=price_per_year.Mean,
                    mode='markers',
                    name='Mean Price'))

fig.add_trace(go.Scatter(x=index, y=price_per_year.Std,
                    mode='markers',
                    name='Std Price'))

fig.update_layout(title="Average MSRP per Year",
                  xaxis_title="Year",
                  yaxis_title="MSRP")

fig.show()


np.argmax(df.loc[df['Year']==2008, 'MSRP'])
data_2008 = df[df['Year']==2008]
data_2008.iloc[326, :]

df_below_2000_filtered = df.loc[((df['Year']<=2000) & (df['MSRP']< 10000))]
fig = px.box(df_below_2000_filtered, x="Year", y="MSRP")

df_below_2000 = df[df['Year']<=2000]

# Plot
fig = px.box(df_below_2000, x="Year", y="MSRP")

reference_line = go.Scatter(x=[1989, 2001],
                            y=[10000, 10000],
                            mode="lines",
                            line=go.scatter.Line(color="red"),
                            showlegend=False)

fig.add_trace(reference_line)

fig.update_layout(title="Boxplots of MSRP per Year for cars sold before 2000",
                  xaxis_title="Year",
                  yaxis_title="MSRP")

fig.show()

df_below_2000_filtered = df.loc[((df['Year']<=2000) & (df['MSRP']< 10000))]

fig = px.box(df_below_2000_filtered, x="Year", y="MSRP")

fig.update_layout(title="Boxplots of MSRP per Year Filtered",
                  xaxis_title="Year",
                  yaxis_title="MSRP")

fig.show()

df_after_2000 = df[df['Year']>2000]

fig = px.box(df_after_2000, x="Year", y="MSRP")

reference_line = go.Scatter(x=[2000, 2018],
                            y=[500000, 500000],
                            mode="lines",
                            line=go.scatter.Line(color="red"),
                            showlegend=False)

fig.add_trace(reference_line)

fig.update_layout(title="Boxplots of MSRP per Year",
                  xaxis_title="Year",
                  yaxis_title="MSRP")

fig.show()

df_after_2000_filtered = df.loc[((df['Year']>2000) & (df['MSRP']< 500000))]

fig = px.box(df_after_2000_filtered, x="Year", y="MSRP")

fig.update_layout(title="Boxplots of MSRP per Year Filtered",
                  xaxis_title="Year",
                  yaxis_title="MSRP")

fig.show()

dic = {1990+i : sum(df['Year']==1990+i) for i in range(28)}
x_dic = [1990 + i for i in range(28)]
y_dic = [dic[1990 + i] for i in range(28)]

# Plot
fig = go.Figure([go.Bar(x=x_dic, y=y_dic)])
fig.show ()
fig.update_layout(title="Car year distribution",
                  xaxis_title="Year",
                  yaxis_title="Count Cars sold")
fig.show()

print("Proportion of observations during last three years:",round(sum(y_dic[-3:])/sum(y_dic),2))

# Percentage of car per brand
counts = df['Make'].value_counts()*100/sum(df['Make'].value_counts())

# 10 most present labels
popular_labels = counts.index[:10]

# Plot
colors = ['lightslategray',] * len(popular_labels)
colors[0] = 'crimson'

fig = go.Figure(data=[go.Bar(
    x=counts[:10],
    y=popular_labels,
    marker_color=colors, # marker color can be a single color value or an iterable
    orientation='h'
)])

fig.update_layout(title_text='Proportion of Car brands in America (in %)',
                  xaxis_title="Percentage",
                  yaxis_title="Car Brand")

print(f"Over {len(counts)} different car brands, the 10 most recurrent car brands in that dataset t represents {np.round(sum(counts[:10]))}% of the total number of cars !")


prices = df[['Make','MSRP']].loc[(df['Make'].isin(popular_labels))].groupby('Make').mean()
print(prices)

# Filtering
data_to_display = df[['Make','Year','MSRP']].loc[(df['Make'].isin(popular_labels)) & (df['Year'] > 2000)]

# Plot
fig = px.box(data_to_display, x="Year", y="MSRP")

fig.update_layout(title="MSRP over the 10 most represented Car brands",
                  xaxis_title="Year",
                  yaxis_title="MSRP")

fig.show()


# Group categories (unleaded, flex-fuel, diesel, electric, natural gas)
df.loc[df['Engine Fuel Type']=='regular unleaded','Engine Fuel Type'] = 'unleaded'
df.loc[df['Engine Fuel Type']=='premium unleaded (required)','Engine Fuel Type'] = 'unleaded'
df.loc[df['Engine Fuel Type']=='premium unleaded (recommended)','Engine Fuel Type'] = 'unleaded'

df.loc[df['Engine Fuel Type']=='flex-fuel (unleaded/E85)','Engine Fuel Type'] = 'flex-fuel'
df.loc[df['Engine Fuel Type']=='flex-fuel (premium unleaded required/E85)','Engine Fuel Type'] = 'flex-fuel'
df.loc[df['Engine Fuel Type']=='flex-fuel (premium unleaded recommended/E85)','Engine Fuel Type'] = 'flex-fuel'
df.loc[df['Engine Fuel Type']=='flex-fuel (unleaded/natural gas)','Engine Fuel Type'] = 'flex-fuel'

eng = df.loc[~df['Year'].isin([2015,2016,2017]),'Engine Fuel Type'].value_counts()
eng2 = df.loc[df['Year'].isin([2015,2016,2017]),'Engine Fuel Type'].value_counts()

print('From last three years: \n')
print(eng, '\n')
print('From 1990 to 2014: \n')
print(eng2)

# Proportion before 2015
prop_eng_ft = pd.DataFrame({'Engine Fuel Type' : eng.index,
                            'Proportion': (eng/sum(eng)).tolist()})

# Proportion after 2015
prop_eng_ft2 = pd.DataFrame({'Engine Fuel Type' : eng2.index,
                            'Proportion 3years': (eng2/sum(eng2)).tolist()})

import plotly
print(plotly.__version__)


# Plot
fig = go.Figure()
fig.show()
fig.add_trace(go.Bar(
    x=prop_eng_ft['Engine Fuel Type'],
    y=prop_eng_ft['Proportion'],
    name='Proportion of cars per fuel type before 2015',
    marker_color='indianred'
))

fig.add_trace(go.Bar(
    x=prop_eng_ft2['Engine Fuel Type'],
    y=prop_eng_ft2['Proportion 3years'],
    name='Proportion of engine fuel type after 2015',
    marker_color='lightsalmon'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  title_text='Proportion of engine fuel type')

fig.show()

# Print correlation matrix
corr = df.corr(numeric_only=True)

# Selecting only numerical features
list_numeric = list(df.describe().columns)

corr = df.corr(numeric_only=True)

# generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype = np.bool_)

# return the indices for the upper triangle of an (n,m) array
mask[np.triu_indices_from(mask)] = True

# Plot
sns.set_style("white")
f, ax = plt.subplots(figsize=(11,7))
plt.title("Correlation matrix")
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220,10, as_cmap=True),
            square=True, vmax = 1, center = 0, linewidths = .5, cbar_kws = {"shrink": .5})

plt.show()

# Plot
fig = px.histogram(df, x="Engine Cylinders", title='Engine cylinders',)
fig.show()

# Get index of highest number of cylinders
index_max_cylinders = df['Engine Cylinders'].idxmax()
index_max_msrp = df['MSRP'].idxmax()
print(index_max_msrp == index_max_cylinders)
print(index_max_msrp == df['Engine HP'].idxmax())

# Get data
data_pie = df['Transmission Type'].value_counts()

# Plot
fig = go.Figure(data=[go.Pie(labels=data_pie.index, values=data_pie.tolist(), textinfo='label+percent',
                             insidetextorientation='radial'
                            )])

fig.update_traces(hole=.3, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Pie chart of transmission type")
fig.show()

# Get data
data_pie = df['Driven_Wheels'].value_counts()

# Plot
fig = go.Figure(data=[go.Pie(labels=data_pie.index, values=data_pie.tolist(), textinfo='label+percent',
                             insidetextorientation='radial'
                            )])

fig.update_traces(hole=.3, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Pie chart of driven wheels variable")

fig.show()
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Handling missing values
df = df.dropna()

# Select features and target variable
features = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity']
X = df[features]
y = df['MSRP']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with preprocessing and the model
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), features)
])
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Prediction for a new car
# Customize these values based on the car you want to predict the price for
new_data = pd.DataFrame
import ui

if __name__ == "__main__":
    ui.create_ui()
predicted_price = model.predict(['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity'])
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your data
df = pd.read_csv("data.csv")

# Preprocessing
# Handling missing values - you might choose to fill them or drop them based on your dataset
df = df.dropna()

# Feature Selection
# Let's say these are your chosen features and target variable
features = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity']
X = df[features]
y = df['MSRP']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Initialization
# Here you can adjust hyperparameters as needed
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define a pipeline with preprocessing and the model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# Model Training
pipeline.fit(X_train, y_train)

# Predict and Evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# ... other imports ...

# Assuming you have a trained model named 'model'
# For example:
# model = RandomForestRegressor()
# model.fit(X_train, y_train)

def predict_price():
    # Assuming you get these values from the user input in your Tkinter app
    year = ui
    engine_hp = ui
    engine_cylinders = ui
    highway_mpg = ui
    city_mpg = ui
    popularity = ui

    # Make sure to reshape the input to 2D array for single prediction
    input_features = np.array([year, engine_hp, engine_cylinders, highway_mpg, city_mpg, popularity]).reshape(1, -1)

    # Use the 'model' to predict
    predicted_price = model.predict(input_features)
    return predicted_price

# In your Tkinter callback
def your_callback_function():
    # ...
    price = predict_price()
    # ...

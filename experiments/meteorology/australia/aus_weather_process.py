import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # Shows all columns

df = pd.read_csv('weatherAUS.csv')

print(df.head(5))

print(len(df))

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Sort by 'Location' and 'Date'
df.sort_values(by=['Location', 'Date'], inplace=True)

# Shift the 'MaxTemp' column to get the temperature of the next day
df['NextDayMaxTemp'] = df.groupby('Location')['MaxTemp'].shift(-1)

# Display the resulting DataFrame
print(df.head())

print(df.columns)
df = df.drop('Date', axis=1)
df = df.drop('RainToday', axis=1)
df = df.drop('RainTomorrow', axis=1)
print(df.columns)

print(len(df))
df = df.dropna()
print(len(df))

print(df.head(5))

print(df['Location'].nunique())
print(df['WindGustDir'].nunique())

# Function to convert wind direction to angle
def direction_to_angle(direction):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return directions.index(direction) * 360 / len(directions)

columns_that_contain_direction = [col for col in df.columns if 'Dir' in col]

for col in columns_that_contain_direction:
    radians = np.deg2rad(df[col].apply(direction_to_angle))
    df[col + '_x'] = np.cos(radians)
    df[col + '_y'] = np.sin(radians)
    # drop the original column
    df = df.drop(col, axis=1)

print(df.columns)

print(df.head(5))

from sklearn.preprocessing import LabelEncoder

df["Location"] = LabelEncoder().fit_transform(df["Location"])

# print(df.head(5))
print(df["Location"].nunique())

# Save the dataframe to a csv file
df.to_csv('weatherAUS_processed.csv', index=False)

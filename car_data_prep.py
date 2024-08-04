#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from datetime import datetime
    from scipy.stats import chi2_contingency
    import matplotlib.ticker as ticker
    from sklearn.linear_model import LassoCV, Lasso, LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import subprocess
    import sys
    from scipy.stats import linregress

    df = df.copy()
       
    duplicates = df[df.duplicated()]
    df = df.drop_duplicates()
    

    null_percent = df.isnull().mean() * 100
    print('Befor drop:\n',null_percent)
    
    #Remove from the model column if there is a manufacturer in the model name
    for i in range(len(df)):
            if df.iloc[i,0] in df.iloc[i,2]:
                df.iloc[i,2]= df.iloc[i,2].replace(df.iloc[i,0], '')
                
    models_with_multiple_manufacturers = df.groupby('model')['manufactor'].nunique().reset_index()
    models_with_multiple_manufacturers = models_with_multiple_manufacturers[models_with_multiple_manufacturers['manufactor'] > 1]
    models_with_multiple_manufacturers
    

    # Replace 'Lexsus' with 'לקסוס' in the 'manufactor' column
    df.loc[df['manufactor'] == 'Lexsus', 'manufactor'] = 'לקסוס'

    # Filter rows where the 'model' is 'לקסוס IS250'
    lexus_is250 = df.loc[df['model'] == 'לקסוס IS250']

    # Filter rows where the 'model' is '320'
    model_320 = df.loc[df['model'] == '320']
    
    #Checking if there is a model with different names and converting to one name
    def change_if_same(value, substring, replacement):
        return replacement if substring in value else value
        # Replace 'סיוויק' with 'CIVIC' using apply function
    df.loc[df['model'] == 'סיוויק', 'model'] = 'CIVIC'

    # Replace values in 'model' column using np.where and .loc
    df.loc[df['model'].isin(["JAZZ", "ג'אז", 'ג`אז']), 'model'] = 'JAZZ'
    df.loc[df['model'].isin(['אקורד']), 'model'] = 'ACCORD'
    
    #Checking what is the 'type' of each column.
    print(df.dtypes)
    
    #Convert columns to different types.
    df.loc[:, 'manufactor'] = df['manufactor'].astype(str)
    df.loc[:, 'model'] = df['model'].astype(str)

    
    print("Converting Year to numeric...")
    print("Year values before conversion:", df['Year'].tolist())
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    print("Year conversion done. Unique values:", df['Year'].unique())
    
    print("Converting Hand to numeric...")
    df['Hand'] = pd.to_numeric(df['Hand'], errors='coerce')
    print("Hand conversion done. Unique values:", df['Hand'].unique())
    
    print("Converting capacity_Engine to numeric...")
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'].str.replace(',', ''), errors='coerce')
    print("capacity_Engine conversion done. Unique values:", df['capacity_Engine'].unique())
    
    print("Converting Km to numeric...")
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')

    print("Km conversion done. Unique values:", df['Km'].unique())
    
    
    #Checking whether there are rows in which 'engine volume' has an Null value.
    #Filling in the empty values of the engine volume according to the type of model.
    #We decided to fill in the most common value depending on the vehicle model because engine volume does not change continuously.
    
    missing_indices = df[df['capacity_Engine'].isnull()].index
    for idx in missing_indices:
        model_value = df.loc[idx, 'model']
    # Find the mode of 'capacity_Engine' for the same 'model'
        model_mode = df[df['model'] == model_value]['capacity_Engine'].mode()
        if not model_mode.empty:
            df.loc[idx, 'capacity_Engine'] = model_mode.iloc[0]
        else:
        # If no model-specific mode is found, use the overall mode of the column
            overall_mode = df['capacity_Engine'].mode()
            if not overall_mode.empty:
                df.loc[idx, 'capacity_Engine'] = overall_mode.iloc[0]
        # Check if there are still any missing values

    missing_values_count = df['capacity_Engine'].isnull().sum()
    df['capacity_Engine'].isnull().sum()
    
    #Calculates the mode (most frequent value) of 'Engine_type' for each combination of 'Year', 'model', and 'manufactor' in the DataFrame.
    #Fill NaN values in Engine_Type column with the mode values of their respective groups
    #Fill remaining NaN values with the overall mode
    #We noticed that there are two engine types with the same meaning.
    mode_values = df.groupby(['Year', 'model', 'manufactor'])['Engine_type'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    df.loc[:, 'Engine_type'] = df['Engine_type'].fillna(mode_values)
    overall_mode = df['Engine_type'].mode()[0]
    df.loc[:, 'Engine_type'] = df['Engine_type'].fillna(overall_mode)
    df['Engine_type'].value_counts()
    df.loc[:, 'Engine_type'] = df['Engine_type'].replace('היבריד', 'היברידי')
    df.loc[:, 'Gear'] = df['Gear'].replace('אוטומט', 'אוטומטית')
    missing_values_count = df['Engine_type'].isnull().sum()    
    missing_values_count
    df['Gear'].unique()

    #that we performed a missing values check by percentage. 
    #'Gear' was less than a percentage of missing values, so we filled in the most common value
    
    df[df['Gear'].isna()]
    overall_Gear = df['Gear'].mode()[0]
    df.loc[:, 'Gear'] = df['Gear'].fillna(overall_Gear)
    
    df['Area'].unique()

    locations = ['רעננה - כפר סבא', 'מושבים בשרון', 'רמת', 'נס ציונה - רחובות',
             'ראשל"צ והסביבה', 'פתח תקוה והסביבה', 'nan', 'חיפה וחוף הכרמל',
             'חולון - בת ים', 'ירושלים והסביבה', 'מושבים', 'כרמיאל והסביבה',
             'באר שבע והסביבה', 'גליל ועמקים', 'עכו - נהריה', 'בית שמש והסביבה',
             'גדרה יבנה והסביבה', 'אשדוד - אשקלון', 'טבריה והסביבה',
             'רמת גן - גבעתיים', 'קריות', 'תל אביב', 'ראש העין והסביבה',
             'עמק יזרעאל', 'נתניה והסביבה', 'בקעת אונו', 'גליל',
             'מודיעין והסביבה', 'פרדס', 'חדרה וישובי עמק חפר',
             'הוד השרון והסביבה', 'רמת השרון - הרצליה', 'None', 'חולון',
             'אזור השרון והסביבה', 'מושבים במרכז', 'קיסריה והסביבה',
             'מושבים בשפלה', 'רעננה', 'טבריה', 'אילת והערבה', 'זכרון - בנימינה',
             'רמלה - לוד', 'תל', 'הוד', 'עמק', 'ירושלים', 'יישובי השומרון',
             'פרדס חנה - כרכור', 'פתח', 'מודיעין', 'רמלה', 'ראשל"צ', 'נתניה',
             'מושבים בצפון', 'ראש', 'נס', 'חיפה', 'מושבים בדרום', 'רחובות']
    # Define the mapping function
    def map_location_to_region(location):
        regions = {
            'צפון': ['חיפה וחוף הכרמל', 'כרמיאל והסביבה', 'גליל ועמקים', 'עכו - נהריה',
                   'טבריה והסביבה', 'קריות', 'גליל', 'טבריה', 'זכרון - בנימינה',
                   'יישובי השומרון', 'פרדס חנה - כרכור', 'חדרה וישובי עמק חפר',
                   'מושבים בצפון'],
            'מרכז': ['רעננה - כפר סבא', 'מושבים בשרון', 'רמת', 'נס ציונה - רחובות',
                   'ראשל"צ והסביבה', 'פתח תקוה והסביבה', 'חולון - בת ים',
                   'ירושלים והסביבה', 'מושבים', 'גדרה יבנה והסביבה', 'רמת גן - גבעתיים',
                   'תל אביב', 'ראש העין והסביבה', 'נתניה והסביבה', 'בקעת אונו',
                   'מודיעין והסביבה', 'פרדס', 'הוד השרון והסביבה', 'רמת השרון - הרצליה',
                   'חולון', 'אזור השרון והסביבה', 'מושבים במרכז', 'קיסריה והסביבה',
                   'רעננה', 'רמלה - לוד', 'תל', 'הוד', 'עמק', 'ירושלים', 'פתח',
                   'מודיעין', 'רמלה', 'ראשל"צ', 'נתניה', 'רחובות'],
            'דרום': ['באר שבע והסביבה', 'אשדוד - אשקלון', 'אילת והערבה', 'מושבים בדרום'],
            'לא ידוע': ['nan', 'None']
    }
        for region, locations_in_region in regions.items():
            if location in locations_in_region:
                return region

        return 'מזרח'  # הנחה כללית

    # Assuming df already exists and has an 'Area' column
    df.loc[:, 'Region'] = df['Area'].apply(map_location_to_region)
    cols_to_drop = ['Area']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    #We noticed that there are illogical values ​​of kilometers compared to a year. For example, in 1983 there is a car that traveled only 100 kilometers.
    #Let's start from the end of the code, we wanted to check if there is a linear relationship between the number of kilometers and the year of manufacture of the vehicle.
    #We realized that there is indeed a distinct statistical relationship.(p value is very small.)
    #We calculated the median value of kilometers by year .
    #We filled in the missing values according to the calculation
    
    # Calculate median values of 'Km' by 'Year'
    median_values = df.groupby('Year')['Km'].median().round(0)
    # Fill NaN values with median values
    df['Km'] = df.apply(lambda row: int(median_values[row['Year']]) if np.isnan(row['Km']) else int(row['Km']), axis=1)
    # Convert 'Km' back to numeric
    slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['Km'])
    print(f"Slope: {slope}, Intercept: {intercept}")
    print(f"R-squared: {r_value**2}")
    print(f"P-value: {p_value}")
    
    #Following the relationship we found between km and years,
    #we decided to perform a linear regression to change the exceptions in the best way
    #We predict the problematic values.
    # Prepare data for linear regression
    X = df[['Year']]
    y = df['Km']
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    # Predict Km using the linear model
    df['Predicted_Km'] = model.predict(X)
    # Calculate residuals (errors)
    df['Residuals'] = df['Km'] - df['Predicted_Km']
    # Calculate standard deviation of residuals
    std_dev = df['Residuals'].std()
    # Define threshold for identifying outliers (e.g., 2 standard deviations)
    threshold = 2 * std_dev
    # Identify outliers
    df['Outlier'] = (np.abs(df['Residuals']) > threshold)
    # Save outliers to a separate DataFrame for adjustment
    outliers_df = df[df['Outlier']]
    # Adjust outliers by replacing them with predicted values from regression in the original DataFrame
    df.loc[outliers_df.index, 'Km'] = df.loc[outliers_df.index, 'Predicted_Km']
    # Drop temporary columns

    
    #Checking whether all the exceptions have changed
    df[df['Outlier']==True] 
    
    #delete the columns that are not relevant as a way to continue the project.
    cols_to_drop = ['Predicted_Km', 'Residuals', 'Outlier','Pic_num','Supply_score','Test','Prev_ownership','Curr_ownership','Color','Description','Repub_date','Cre_date']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    

    
    
    #We wanted to check if there is a statistical relationship
    #between the number of hands, and the price of the vehicle.
    #You can see from the graph that there is a relationship between hand and price.
    #When the number of hands increases, the price of the vehicle becomes smaller.
    

    
    #We performed another test using linear regression. The test showed that there is indeed a strong relationship between price and the amount of hands. 
    #It can also be seen in the graph and in the value of P.
    X = df[['Hand']]  # Change to 2D array
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)
    # Step 2: Get the slope, intercept, R-squared, and p-value
    slope, intercept, r_value, p_value, std_err = linregress(df['Hand'], df['Price'])
    
    
   
    X = df[['Year']]  # Change to 2D array
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)
    # Step 2: Get the slope, intercept, R-squared, and p-value
    slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['Price'])

    
    #Checking how the engine volume affects the price.
    #First we checked what the most common engine volume values are.
    counts = df['capacity_Engine'].value_counts()
    counts.head(10)
    
    #We used boxplot to try to understand how the engine volume affects the price.
    top_values = df['capacity_Engine'].value_counts().nlargest(7).index
    # Step 2: Filter data and calculate median price
    filtered_df = df[df['capacity_Engine'].isin(top_values)]
    median_prices = filtered_df.groupby(['capacity_Engine','Year'])['Price'].median().reset_index()
    # Sort top_values by capacity_Engine
    top_values_sorted = sorted(top_values, key=float)  # Sort as floats if they are numeric strings
    # Step 3: Create box plot
 
    
    top_5_engines = df['capacity_Engine'].value_counts().nlargest(5).index.tolist()
    print(f'Top 5 Popular Capacity Engines: {top_5_engines}')
    # Filtering the data to contain only the five popular volumes
    filtered_df = df[df['capacity_Engine'].isin(top_5_engines)]
    # Receiving quantities by year and engine volume
    capacity_by_year = filtered_df.groupby(['Year', 'capacity_Engine']).size().reset_index(name='Count')
    # Function to find the difference between the quantities
    def calculate_year_diff(group):
        if len(group) < 5:
            return np.nan
        sorted_counts = sorted(group['Count'])
        diffs = [abs(sorted_counts[i] - sorted_counts[i + 1]) for i in range(len(sorted_counts) - 1)]
        return sum(diffs) / len(diffs) if diffs else np.nan
    # Calculation of the difference by year
    year_diffs = capacity_by_year.groupby('Year').apply(calculate_year_diff).reset_index(name='Diff')
    # Finding the three years that are closest in quantity
    closest_years = year_diffs.dropna().nsmallest(3, 'Diff')['Year'].tolist()
    # Guaranteed at least 20 data in each selected year
    def check_min_count(year, min_count=20):
        return len(df[df['Year'] == year]) >= min_count
    # Cutting the data according to the years found and performing groupby
    closest_years = [year for year in closest_years if check_min_count(year)]
    print(f'Closest Years with at least 20 entries: {closest_years}')
    final_filtered_df = df[df['Year'].isin(closest_years)]
    grouped_filtered_df = final_filtered_df.groupby(['Year', 'capacity_Engine']).size().reset_index(name='Count')
    print(grouped_filtered_df)
    
    # Ensure 'Price' is not missing and is numeric
    if 'Price' not in df.columns:
        raise ValueError("Column 'Price' is missing from the DataFrame")

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')    

    
    
    
    
    features = ['manufactor', 'model', 'Year', 'capacity_Engine', 'Engine_type','Price']
    df = df[features]
#     except KeyError:
#         features = ['manufactor', 'model', 'Year', 'capacity_Engine', 'Engine_type']
#         df = df[features]
#     print('Features selected.')
    
    

    
    df = pd.get_dummies(df)
    
#     # בדיקה אם עמודת Price קיימת
#     if 'Price' in df.columns:
#         # שמירת עמודת Price בנפרד
#         price = df['Price']
#         df = df.drop(columns=['Price'])
#         df['Price'] = price
#         print("Price column moved to the end.")
    
#     print("Data preparation finished.")
    return df


# In[ ]:





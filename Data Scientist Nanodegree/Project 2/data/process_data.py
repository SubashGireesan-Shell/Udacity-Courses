'''
Udacity Nanodegree - Project 2
Disaster Response Pipeline Project

ETL Pipeline

Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Arguments:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)
'''

# Imports
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Loading Data
def load_data(messages_filepath, categories_filepath):
    '''
    Function to load messages and categories data
    
    Arguments:
        messages_filepath - File path to messages csv file
        categories_filepath - File path to categories csv file
    Returns:
        df - Merged dataframe with messages and categories data
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df

#  Cleaning data
def clean_data(df):
    '''
    Function to clean the dataframe
    
    Arguments: 
        df - Pandas dataframe with raw data
    Returns:
        df - Cleaned pandas dataframe
    '''

    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
        categories[column] = categories[column].replace(2, 0)
    df = df.drop('categories', axis=1)
    df = pd.concat([df,categories], axis=1)
    df = df.drop_duplicates()
    return df

# Saving data
def save_data(df, database_filename):
    '''
    Function to save data
    
    Arguments:
        df - Dataframe object
        database_filename - Path to database file
    Returns:
        None
    '''

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False, if_exists='replace')
    pass  


# Main function
def main():
    '''
    Function implementing the ETL Pipeline:
        - Data Extraction
        - Data Transformation (Cleaning)
        - Data Loading
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # messages_filepath = 'disaster_messages.csv'
        # categories_filepath = 'disaster_categories.csv'
        # database_filepath = 'YourDatabaseName.db'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
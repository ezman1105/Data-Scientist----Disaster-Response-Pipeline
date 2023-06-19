# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


def load_data(messages_filepath, categories_filepath):
    
    """
    load_data --> function of loading and merging datasets
    
    Arguments:
    
    messages_filepath --> a string of filepath for a csv file
    
    categories_filepath --> a string of filepath for a csv file
    
    Output:
    
    df --> a merged raw dataframe combining messages and categories
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner') # Merge the datasets
    
    return df


def clean_data(df):
    
    """
    clean_data --> function of processing of wranggling
    
    Argument:
    
    df --> a combined dataset including messages and categories
    
    Output:
    
    df --> a cleaned combined dataset including messages and categories
    
    """
    
    categories = df['categories'].str.split(';', expand=True) # create a dataframe of the 36 individual category columns
    
    # Use the first row of categories dataframe to create column names for the categories data.
    
    row = categories.iloc[0] 
    cat_colname = row.str.replace('[-,1,0]', '')
    categories.columns = cat_colname # # Rename the columns of `categories`
 
    # Convert content into '0' or '1'
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)

    # Remove unnecessary column 'categories', and concat new dataframe 'cat_split' and original dataframe 'df'
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)
    
    # Remove unnecessary column 'child_alone', wrongly labeled value '2' and duplicates
    df = df.drop('child_alone', axis=1)
    df = df[df["related"] < 2 ] # "2" are mistakely included in column 'related', which should be removed
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    
    """
    save_data --> function of saving dataframe into sql light database
    
    Argument:
    
    df --> a cleaned combined dataset including messages and categories
    
    database_filename --> a filename of a SQLite database 
       
    """
    
    database_filename = 'DisasterResponse.db'
    conn = sqlite3.connect(database_filename)
    df.to_sql('table1', conn, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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
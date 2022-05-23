# import libraries
import sys
import logging

import pandas as pd
from sqlalchemy import create_engine

# configuration of the logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_data(messages_filepath, categories_filepath):
    """ Loads the data from the filepaths aguments, and merge the data into a pandas dataframe.
    
    Args:
        messages_filepath (string): Filepath of the messages data.
        categories_filepath (string): Filepath of the categories data.
    
    Returns:
        pandas dataframe. It merges messages and categories data.
    
    """
    
    try:
        # load messages dataset
        messages = pd.read_csv(messages_filepath)
        
        # load categories dataset
        categories = pd.read_csv(categories_filepath)
        
    except Exception as err:
        raise error
                 
    else:
        # return a dataframe that merges the two datasets
        df = messages.merge(categories, how='inner')
        
        logging.info('Data loading process finished.')
        
        return df


def clean_data(df):
    """ Cleans the data in the pandas dataframe argument.
    
    Cleaning process comprise:
    1. Cleaning of the category names.
    2. Cleaning of the category values, and casting to integer data type.
    3. Removal of duplicates messages.
    
    Args:
        df -- pandas dataframe with the data to clean.
    
    Returns:
        pandas dataframe. Cleaned data.
    
    """
    
    try:
        # create a dataframe of the 36 individual category columns
        categories = df.categories.str.split(';', expand=True)

        # replace the columns of the 'categories' dataframe by a list of streamlined category names
        categories.columns = [value.split('-')[0] for value in categories.iloc[0]]
    
        # set each category value to be the last character of the string, casted to integer
        for column in categories:
            categories[column] = categories[column].apply(lambda x: int(x[len(x) - 1])) 
    
        # drop the original categories column from `df`
        df.drop('categories', axis=1, inplace=True)
    
        # concatenate the original dataframe with the new `categories` dataframe
        df = df.merge(categories, how='inner', left_index=True, right_index=True)
    
        # drop duplicates
        df.drop_duplicates(subset=['message'], inplace=True)
        
        # drop rows with value '2' in column 'related'
        df.drop(df[df.related==2].index, axis=0, inplace=True)
        
    except Exception as err:
        raise error
    
    else:
        # return the cleaned dataframe
        logging.info('Data cleaning process finished.')
        
        return df


def save_data(df, database_filename):
    """ Store the data in a SQL database.
    
    Args.
        df (pandas dataframe) -- Data to store.
        database_filename (string) -- Filepath of the database the data has to be stored in.
        
    """
    
    try:
        engine = create_engine(f'sqlite:///{database_filename}')
        df.to_sql('data', engine, if_exists='replace', index=False) 
    
    except Exception as err:
        raise err
    
    else:
        logging.info('Data storing process finished.')
        
        

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
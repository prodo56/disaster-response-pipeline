import sys
import pandas as pd
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    '''
    This function reads the input files and merges them based on the ID column 
    present in the 2 files
    
    :param messages_filepath: path to file containing messages
    :param categories_filepath: path to file containing categories of messages
    
    :return: DataFrame object with both the files merged
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets based on the `id` column
    df = messages.merge(categories,on=["id"],how="inner")
    
    return df



def clean_data(df):
    '''
    This functions cleans the DataFrame by dropping the duplicates,
    exapnding the categories to separate columns and converting them to 
    appropriate data types
     
    :param df: DataFrame object to be cleaned
    
    :return: `df` cleaned DataFrame object
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda col: col.split("-")[0],row))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(["categories"],axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # check number of duplicates
    duplicates = df[df.duplicated()].shape[0]
    
    print("Found {} duplicate row(s). Deleting them..".format(duplicates))
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
#     df.message.str.encode('utf-8', errors='strict')
    
    return df
        


def save_data(df, database_filename):
    '''
    Saves the given DataFrame to db

    :param df: DataFrame object to be saved
    :param database_filename: name of the db to be created
    '''
    if os.path.exists(database_filename):
        os.remove(database_filename)
    #create db engine object
    engine = create_engine('sqlite:///'+database_filename)
#     engine.raw_connection().connection.text_factory = str
    
    #store the df to db
    df.to_sql("clean_data", engine, index=False)  


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
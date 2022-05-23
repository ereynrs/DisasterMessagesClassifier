# import libraries
import sys
import re
import logging
import pickle

import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# configuration of the logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_data(database_filepath):
    """ Loads the data from the database filepath argument.
    
    Args:
        database_filepath (string): Filepath of the database.
        
    Returns:
        array. Features data.
        array. Target data.
        array. Class labels.
    
    """
    
    logging.info('Data loading process started.')
    
    try:
        # load data from the database
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('data', engine)
        
        # retrieve features, targets, and labels
        X = df.message.values
        y = df.iloc[:, 4:].values
        category_names = df.iloc[:, 4:].columns
        logging.info('Features and classes splitted.')
    
    except Exception as err:
        raise err
        
    else:
        logging.info('Data loading process finished.')
        
        # return features and classes
        return X, y, category_names
        

def tokenize(text):
    """ Tokenize the text argument.
    
    Tokenization process comprise:
    1. URLs replaced by the 'urlplaceholder' word. 
    2. Word-based tokenization.
    3. Token lowercased.
    3. Leading ans trailing spaces removed.
    
    Args:
        text (string): Text to tokenize.
        
    Returns:
        list. Cleaned tokens.
    
    """
    
    # regex to detect a url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # tokenize the text argument
    # urls are replaced by the 'urlplaceholder' word
    # tokens are lowercased
    # leading and trailing characters are removed
    tokens = [tok.lower().strip() for tok in word_tokenize(re.sub(url_regex, 'urlplaceholder', text))]
    
    # initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemm_toks = [lemmatizer.lemmatize(tok) for tok in tokens]

    # return clean and lemmatized tokens
    return lemm_toks 


def build_model():
    """ Build a model as a pipeline.
    
    The pipeline comprise:
    1. A Count Vectorizer transformer.
    2. A TF-IDF transformer.
    3. A MultiOutputClassifier applying a Decision Tree Classifier as estimator for each class.
    4. The parameters to perform a grid search are established.
    
    Returns:
        object. A parametrized grid search object.
    
    """
    
    logging.info('Model building process started.')
    
    # creates the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=DecisionTreeClassifier()))
    ])
    
    # specify the gris search parameters
    parameters = { 
        'tfidf__smooth_idf': [False, True],
        'clf__estimator__splitter': ['best', 'random']
    }
    
    # create the model
    cv = GridSearchCV(pipeline, param_grid=parameters)
    logging.info('Model building process finished.')
    
    # create and return the grid objects
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate a model on the test set arguments.
    
    Args:
        model (object): Model to evaluate.
        X_test (array): Features test set.
        Y_test (array): Target test set.
        category_names (list): Names of the categories to classify the data in.
    
    """
    
    # get predictions
    y_pred = model.predict(X_test)
    
    # print the classification report for each category
    for i, category in enumerate(category_names):
        print(f'CLASSIFICATION REPORT FOR: "{category} - {i + 1} of {len(category_names)}"')
        print(classification_report(Y_test[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """ Save the model as a pickle file.
    
    Args:
        model (object): Model to save.
        model_filepath (string): Filepath of the model.
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
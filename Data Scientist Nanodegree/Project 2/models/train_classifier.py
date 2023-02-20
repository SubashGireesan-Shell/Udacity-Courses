'''
Udacity Nanodegree - Project 2
Disaster Response Pipeline Project

ML Pipeline

Sample Script Execution:
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
'''
# Imports
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import fbeta_score, classification_report
from scipy.stats.mstats import gmean
from sklearn.metrics import confusion_matrix

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('omw-1.4')

# Loading data
def load_data(database_filepath):
    '''
    Function to load data from database
    
    Arguments:
        database_filepath: Path to SQL database
    Returns:
        X - Feature dataframe
        y - Label dataframe
        category_names - List of unique categories
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

# Tokenizing text
def tokenize(text):
    '''
    Function for tokenizing the text data

    Arguments:
        text - List of text messages that are to be tokenized
    Returns:
        clean_tokens - List of cleaned tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# User-defined class
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Class for extracting starting verb of a sentence
    '''

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# Building model
def build_model():
    '''
    Function for building model

    Argument -  
        None
    Returns - 
        A Scikit ML Pipeline that process text messages and apply a classifier
    '''
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return model

# Evaluate model
def evaluate_model(model, X_test, y_test, category_names):
    '''
    Function to evaluate the model

    Arguments: 
        model - ML pipeline
        X_test - Test features
        y_test - Test labels
        category_names - Names of the labels
    Returns:
        None
    '''
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'========================{i, col}========================')
        print(classification_report(list(y_test.values[:, i]), list(y_pred[:, i])))

# Save model
def save_model(model, model_filepath):
    '''
    Function to save a trained model as Pickle file, to be loaded later.
    
    Arguments:
        model - GridSearchCV or Scikit Pipelin object
        model_filepath - Destination path to save .pkl file
    Returns:
        None    
    '''
        
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass

# ML Pipeline Main Function
def main():
    '''
    Main function for the ML Pipeline involving:
        1) Extracting data from SQLite db
        2) Training ML model on training set
        3) Estimating model performance on test set
        4) Saving the trained model as Pickle
    '''
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
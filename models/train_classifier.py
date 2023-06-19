# import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sqlite3
from sqlalchemy import create_engine

from sklearn.svm import SVC
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,make_scorer

import warnings # suppress warnings from final output
warnings.simplefilter("ignore")


def load_data(database_filepath):
    
    """
    load_data --> function of loading database 
    
    Argument
    
     --> filepath of the database
    
    Output: 
    X --> dataframe of feature ('message')
    
    y --> dataframe of labels
    
    """
    conn = sqlite3.connect(database_filepath)
    #engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql_query('SELECT * FROM table1', conn)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names
        
    

def tokenize(text):
    """Tokenization function: 
    Input: raw text 
    
    Process: 
    url replacement
    normalized
    stop words removed
    lemmatized
    
    Output: tokenized text"""
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # replace url with "urlplaceholder"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize
    words = word_tokenize (text)
    
    
    # Remove Stopwords
    stop_words = stopwords.words("english")
    words = [w for w in words if w not in stop_words]
    
    #lemmatizing
    clean = [WordNetLemmatizer().lemmatize(w, pos='n').strip() for w in words]
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos='v').strip() for w in clean]
   
    return clean_tokens


def build_model():
    
    """
    build_model --> function of creating model for prediction
    
    Output:
    
    cv --> classification model 
    
    """
    
    # create a pipeline
    pipeline_svm = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf_svm',  MultiOutputClassifier(SVC(max_iter=100000, random_state=42))) 
    ])
    
    parameters_svm= {'clf_svm__estimator__C': [0.1, 1,],
                 'clf_svm__estimator__gamma': [1, 0.1]
                    }
    
    cv = GridSearchCV(pipeline_svm, param_grid=parameters_svm, cv=2, verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    evaluate_model --> function of evaluating the model
    
    Argument:
    
    model --> the built classification model
    
    X_test --> features of test set
    
    Y_test --> labels of target set
    
    category_names --> names of categories
    
    """
    
    y_pred_svm = model.predict(X_test)
    
    # Classification report
    print("Classification report")
    print(classification_report(Y_test, y_pred_svm, target_names = category_names))
    


def save_model(model, model_filepath):
    
    """
    save_model --> function to save the model
    
    Argument:
    
    model --> the build classification model
    
    model_filepath --> route of model
    
    """
      
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    


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

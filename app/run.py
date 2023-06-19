import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import re
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from sqlalchemy import create_engine
import sqlite3



app = Flask(__name__)

def tokenize(text):
    
    """
    tokenize --> process of normalizing, tokenizing, stemming, lematizing, replacing url path, and removing stop words
    
    Argument:
    
    text --> raw text messages
    
    Output:
    
    clean_tokens --> list of cleaned tokens
    
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # replace url with "urlplaceholder"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize (text)
    
    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    #lemmatizing
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return clean_tokens


# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('table1', engine)

#con = sqlite3.connect('sqlite:///../data/DisaterResponse.db')
#df = pd.read_sql_query("SELECT * FROM table1", con)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data for visualization
    # TODO: Below is an example - modify to create your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    
    #################################
    
    # Visualize -- top 10 categories
    
    top_cat_number = df.iloc[:,4:].sum().sort_values(ascending=False)[1:11]
    top_colname = list(top_cat_number.index)
    
    #################################
    
    # Visualize -- aid_related status by "0" & "1"
    
    aid_rel_1 = df[df['aid_related']==1].groupby('genre').count()['message']
    aid_rel_0 = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_name = list(aid_rel_1.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_colname ,
                    y=top_cat_number
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        #######################
        {
            'data': [
                Bar(
                    x=genre_name ,
                    y=aid_rel_1,
                    name = "Aid Related"
                ),
                Bar (x=genre_name ,
                    y=aid_rel_0,
                    name = "Aid Not Related"
                )
               
            ],

            'layout': {
                'title': 'Distribution of Messages and Genre by "Air-Related"',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Genre"
                },
                
                'barmode' : 'group'
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
import json
import plotly
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract the target data
    categories = df.iloc[:, 4:]
    # counts the number of occurrences (value != 0) per category
    categories_counts = categories.apply(lambda x: sum(x != 0)).values
    # get the category names
    categories_names = categories.columns.values
    
    # matrix of co-ocurrences of classes.
    # i.e.: number of messages classified both in row `i` and column `j`categories 
    cooccurr_matrix = np.dot(categories.values.T, categories.values)
    
    # coocur matrix is scaled according to the number of messages clasiffied in the row `i`
    cooccurr_percentage = np.zeros((36,36))
    for i, _ in enumerate(categories_counts):
        # for each category, 
        # if the number of messages in the category is zero, 
        # then co-occurence is zero as well
        if categories_counts[i] == 0:
            cooccurr_percentage[i, :] = 0
        else:
            # in other case, calculate the percentage of co-ocurrences
            # and the round to the second digital place
            cooccurr_percentage[i, :] = np.round(((cooccurr_matrix[i, :] / categories_counts[i]) * 100), 2)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],
            
            'layout': {
                'title': 'Number of messages per category',
                'yaxis': {
                    'title': "Count",
                },
                'xaxis': {
                    'tickangle': -30
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=cooccurr_percentage,
                    x=categories_names,
                    y=categories_names
                )
            ],
            
            'layout': {
                'title': 'Percentage of messages in category `x` also in category `y`',
                'yaxis': {
                    'visible': False
                }, 
                'xaxis': {
                    'visible': False
                }
            }
        }
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
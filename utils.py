import json
import json
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import itertools
import plotly.graph_objects as go
import pandas as pd
from constant import *
import plotly.graph_objs as go
import config
from sentence_transformers import SentenceTransformer
import yake

config.kw_model =  yake.KeywordExtractor(top=10,n=1, stopwords=None)
config.sent_model = SentenceTransformer('all-MiniLM-L6-v2')

layout = go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0  #top margin
    )

def stripper(x):
    try:
        if x>=1000:
            return str(x//1000) + 'k+'
        else:
            return str(x)
    except:
        return x

def data_load(file):
    data = json.loads(file)
    df = pd.DataFrame(data['payload']['value'])
    df = df[columns]
    df.columns = formatted_columns
    config.df = df
    config.data = data


def create_table(temp,text, show_index=True):
    temp.reset_index(inplace=show_index)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(temp.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[temp[x] for x in temp.columns],
                   fill_color='lavender',
                   align='left'))
    ], layout_title_text=text)
    fig.update_layout(margin=layout)
    return fig

def comparison(x,key1,key2,criteria):
    if x>0:
        if criteria not in ['read_%','stop word %']:
            return '{} has {} more {}'.format(key1,x,criteria,key2)
        else:
            return "{} has {} % higher {} ".format(key1,x,criteria,key2)
    
    elif x<0:
        if criteria not in ['read_%','stop words %']:
            return '{} has {} more {}'.format(key2,abs(x),criteria)
        else:
            return "{} has {} % higher {} ".format(key2,abs(x),criteria)
    
    return 'no difference'

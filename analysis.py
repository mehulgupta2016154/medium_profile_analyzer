
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm
import itertools
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.cluster import MeanShift
from constant import *
from PIL import Image
from utils import *
import random
import calendar
from transformers import pipeline 

def summary(df):
    data = {}
    cols = st.sidebar.columns(2)
    cols[0].metric('total views',stripper(df['views'].sum()))
    cols[1].metric('total reads',stripper(df['reads'].sum()))

    cols = st.sidebar.columns(2)
    cols[0].metric('total blogs',len(df))
    cols[1].metric('last post',str(df['first Published At'].max())[:7])

def blog_percent(df):
            st.subheader('Total blogs %')
            st.text('this section provides information around blogs satisfying below criteria ')
            for x in ['views','read_%','claps','reading Length (minutes)']:
                st.markdown("<b><u>{}</b></u>".format(x),unsafe_allow_html=True)

                cols = st.columns(3)
                total_blogs = len(df)
                
                for y in range(3):
                    value = str(round((100*len(df[df[x]>=metric_dict[x][y]]))/total_blogs,1))
                    cols[y].metric('{} {}'.format(stripper(metric_dict[x][y]),x) ,'{}%'.format(value))




def top_performers(df):
    top = {}
    temp = df.set_index('title')

    temp = temp.select_dtypes(include=['float64','int64','float32','float32'])

    for x in temp.columns:
        top.update({x:[[x,str(y)] for x,y in temp[x].sort_values(ascending=False).head(1).iteritems()]})
    
    temp = pd.DataFrame([x[0] for x in top.values()],index=[x for x in top.keys()],columns=['blog','value'])
    st.subheader('Top blogs')
    st.text('Highest values given a specific category')
    temp =  temp.loc[~temp.index.isin(ignore),:]
    st.dataframe(temp)

def describe(df):
    temp = df.loc[:,~df.columns.isin(ignore)]
    remainder = lambda x : 1 if x%3>0 else 0
    total_cols = len(temp.columns)//3 + remainder(len(temp.columns))
    columns = iter([x for x in temp.columns])
    for x in range(total_cols):
        cols = st.columns(3)
        for y in range(3):
            try:
                fig = px.box(temp, y=next(columns),color_discrete_sequence=random.choices(px.colors.qualitative.Alphabet,k=1))
                fig.update_layout(margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0  #top margin
    ))
                cols[y].plotly_chart(fig,use_container_width=True)
            except:
                break


def blog_frequency(df):
    agg = st.radio('',('month','year','weekday'))
    if agg=='month':

        temp = df['first Published At Bucket'].value_counts()
        temp.index = pd.to_datetime(temp.index)
        temp.sort_index(inplace=True)
        fig = px.line(x=temp.index, y=temp, title='Blog Frequency',color_discrete_sequence=random.choices(px.colors.qualitative.Alphabet,k=1)).update_layout(xaxis_title="Date", yaxis_title="blog published")
    
    else:
        temp = df[agg].value_counts().sort_index()
        fig = px.line(x=temp.index, y=temp, title='Blog Frequency',color_discrete_sequence=random.choices(px.colors.qualitative.Alphabet,k=1)).update_layout(xaxis_title="Year", yaxis_title="blog published")
        if agg=='weekday':
            fig.update_layout(
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = df['weekday'].unique(),
                    ticktext = [calendar.day_abbr[x] for x in df['weekday'].unique()]
                )
            )
    st.plotly_chart(fig,use_container_width=True)
    
def word_cloud(text,title):
    text_wordcloud = WordCloud(collocations = False, background_color = 'black',width=600, height=300, margin=3).generate(text)
    text_wordcloud.to_file('wc.png')
    st.image(Image.open('wc.png'),caption='wordcloud for {}'.format(title), use_column_width=True)

def distribution_plot(df):
    fig,ax = plt.subplots(4,2,figsize=(20,15))
    for index,title in enumerate(['upvotes', 'views', 'reads','reading Length (minutes)', 'claps','read_%','days since published','avg views per day']):
        row,col = index//2, index%2
        mu, std = norm.fit(df[title])
        ax[row,col].hist(df[title],alpha=0.6, color='blue',bins=20)
    
        xmin, xmax = df[title].min(), df[title].max()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax[row,col].plot(x, p*1000*df[title].max()/100, 'k', linewidth=2)
        ax[row,col].set_xlabel(title)
        ax[row,col].set_ylabel('blog count')
    ax[3,1].set_visible(False)
    fig.suptitle('Distribution plots', fontsize=20)
    st.pyplot(fig,use_container_width=True)

def similar_blogs(df):
    temp = df.drop(['title tokens','first Published At'],axis=1)
    temp = temp.set_index('title')
    temp.index.name = 'title'
    standardized_data =(temp-temp.min())/(temp.max()-temp.min())
    standardized_data.dropna(inplace=True,axis=1)
    meanshift = MeanShift(cluster_all=False)
    temp['label'] = meanshift.fit_predict(standardized_data)
    temp.reset_index(inplace=True)
    user_dict = {label:[] for label in temp['label'].unique()}
    for index, rows in temp.iterrows():
                user_dict.update({rows['label']:user_dict[rows['label']]+[rows['title']]})
    pad = max([len(x) for y,x in user_dict.items()])
    user_dict = {x:y+['' for a in range(pad-len(y))] for x,y in user_dict.items()}
    temp = pd.DataFrame([list(x) for x in list(zip(*user_dict.values()))],columns=['Group '+str(x) for x in temp['label'].unique()])
    fig  = create_table(temp,'similar blogs, performance wise',False)
    fig.update_layout(height=700)
    
    st.plotly_chart(fig,use_container_width=True)

def clustering(df):
    combo = [' & '.join([x[0],x[1]]) for x in list(itertools.combinations(['upvotes','weekdays','days since published', 'views', 'reads','reading Length (minutes)', 'claps','read_%'],2))]
    
    mandatory = ['views & reads','upvotes & claps','upvotes & reading Length (minutes)','days since published & views']
    options = st.multiselect('Add more pairs for clustering',combo)
    options.extend(mandatory)
    options = set(options)
    if options:
            fig,ax = plt.subplots((len(options)//3)+1,3,figsize=(15,3+3*(len(options)//3)))
            for index,pair in enumerate(options):
                    pair = pair.split(' & ')
                    row,col = index//3,index%3
                    temp = df[pair]
                    standardized_data = (temp - temp.min()) / (temp.max() - temp.min())
                    standardized_data.dropna(inplace=True,axis=1)
                    meanshift = MeanShift(cluster_all=True)
                    label = meanshift.fit_predict(standardized_data)
                    u_labels = np.unique(label)
                    for i in u_labels:
                                    ax[row, col].scatter(temp[label == i].loc[:, pair[0]], temp[label == i].loc[:, pair[1]],label='cluster ' + str(i))
                    ax[row,col].set_title(' & '.join(pair))
                    ax[row, col].legend(loc='best')
                    ax[row, col].set_xlabel(pair[0])
                    ax[row, col].set_ylabel(pair[1])
            for remaining in range(index+1,((len(options)//3)+1)*3):
                row,col = remaining//3,remaining%3
                ax[row,col].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig,use_container_width=True)

def correlation(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr = corr.mask(mask)
    fig = go.Figure(data=go.Heatmap(
                    z=corr,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='balance',
                    zmin=-1,
                    zmax=1
                ), layout_title_text='correlation')
    st.plotly_chart(fig,use_container_width=True)

def blog_summary(text):
    summarizer = pipeline("summarization")
    summarized = summarizer(text, min_length=25, max_length=50)
    st.subheader('Summary')
    st.markdown("<i>{}</i>".format(eval(summarized[0])['summarized_text']),unsafe_allow_html=True)

def speedometer(x):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = x*100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "How similar are the 2 blogs?", 'font': {'size': 24}},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "red"},
        'borderwidth': 2,
        'threshold': {
            'line': {'color': "darkblue", 'width': 5},
            'thickness': 0.75,
            'value': 50},
        'bordercolor': "gray"}))

    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
    st.plotly_chart(fig,use_container_width=True)
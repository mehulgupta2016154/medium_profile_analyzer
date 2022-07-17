import sys
sys.path.append("..")

import pandas as pd
from constant import *
from feature_engineering import *
from analysis import *
import streamlit as st
import utils
import json
from io import StringIO
from timeline import *
import config

st.title('Medium Profile Analyser')

try:    
    if len(config.data)>0:
        with st.spinner(text='loading stats'):
            followers(config.data['payload']['paging']['path'])
            
            df = config.df
            summary(df)
            timeliner(df)
            top_performers(df)
            blog_percent(df)
            with st.expander('Blog frequency'):
                blog_frequency(df)
            with st.expander('Word Cloud based on blog titles'):
                st.markdown("<p><b><i>Word clouds</b> work in a simple way: the more a specific word appears in a source of textual data (blog title in our case), the <b>bigger and bolder </b>it appears in the word cloud.</i></p>",unsafe_allow_html=True)
                word_cloud('  '.join(list(itertools.chain(*df['title tokens'].values))),'blog title')
            with st.expander('Correlation heatmap'):
                st.markdown("Depicts how related two variables are: If <ul><li>Closer to 1, variables are +vely correlated i.e. if x increases, y also increases & vice versa</li><li>Closer to -1, variables are -vely correlated i.e. if x increases, y decreases & vice-versa</li><li>Closer to 0, no correlation eists between the variables</li></ul>",unsafe_allow_html=True)
                correlation(df.loc[:,~df.columns.isin(ignore)])
            with st.expander('Clustering'):
                clustering(df)
            with st.expander('Boxplots'):
                st.markdown("Helps to understand distribution of different aspects",unsafe_allow_html=True)
                describe(df)
except:
    st.error('please upload stats file first following About section')







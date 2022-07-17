
from streamlit_timeline import timeline
import streamlit as st
import requests
import re

def timeliner(df):
    options = [str(x) for x in df['year'].unique()]
    options.append('all')
    option = st.radio('choose year for timeline',options,horizontal=True)
    if option!='all':
        df=df[df['year']==eval(option)]
    data = {"events":[]}
    with st.spinner(text="Building timeline for year {}...".format(option)):
        for index,rows in df.iterrows():
            temp = {}
            temp.update({'start_date':{'year':rows['year'],'month':rows['month']},'media':{'url':rows['image Url'],'link':rows['url']},'text':{'headline':rows['title']}})
            data['events'].append(temp)
        timeline(data, height=500)

def followers(user_name):
    try:
        user_name = user_name.split('/')[1]
        page1 = requests.get('https://medium.com/{}/about'.format(user_name))
        followers = re.findall('(\d+) Followers',page1.text)[0]
        following = re.findall('(\d+) Following',page1.text)[0]

        cols = st.sidebar.columns(2)

        cols[0].metric('Followers',followers)
        cols[1].metric('Following',following)
    except:
        pass

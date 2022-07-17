import streamlit as st
from streamlit_player import st_player
from utils import *
from feature_engineering import *


st.set_page_config(page_title='Medium Profile Analyzer' ,layout="wide",page_icon='Ⓜ️')

st.title('Medium Profile Analyzer')
st.text('This webapp helps you to get a hang around your medium profile.\nYou just need to follow the beloe steps to get started')
st.markdown('''<ul><li>Login into your <a href='https://medium.com/'>Medium Account</a></li><li>Go to the <a href='https://medium.com/me/stats?format=json&limit=1000'>link</a></li><li>Save the file using <b>Ctrl+s</b></li><li>Upload the file in <b>My Analysis</b> section</li></ul>''',unsafe_allow_html=True)
file = st.file_uploader('upload your medium stats json')
if file is not None:
    file = file.read()
    file = file[16:]
    data_load(file)
    engineering(config.df,config.data)
st_player("https://www.youtube.com/watch?v=WYxS1FTc4Kw")
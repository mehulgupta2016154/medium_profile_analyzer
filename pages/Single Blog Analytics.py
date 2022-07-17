import sys
sys.path.append("..")

import pandas as pd
from constant import *
from feature_engineering import *
from analysis import *
import streamlit as st
import utils
from timeline import *
from blog_scrapper import *
import config
from scipy import spatial
from fuzzywuzzy import process

try:
    df = config.df
except:
    st.error('please upload stats file first following About section')
try:
        blog = st.selectbox('select blog for analysis',[None]+[x for x in df['title'].values])
        if blog:
            with st.spinner('loading stats ...'):
                text_tokens,stats, original_text = get_blog_text(df[df['title']==blog]['url'].values[0])
                word_cloud(' '.join(text_tokens),'blog')

                temp = {x:[a for a in y.values()][0] for x,y in df[df['title']==blog][['first Published At Bucket','publication','avg views per day','read_%','views','reads','claps','upvotes','reading Length (minutes)']].to_dict().items()}
                stats.update(temp)
                temp = pd.DataFrame(columns=['stat','value'])
                temp['stat'] = [x for x in stats.keys()]
                temp['value'] = [str(x) for x in stats.values()]
                fig = utils.create_table(temp,'blog statistics',False)
                fig.update_layout(height=300)
                st.plotly_chart(fig,use_container_width=True)
                keywords = config.kw_model.extract_keywords(' '.join(text_tokens))
                st.subheader('Keywords')
                cols = st.columns(5)
                seen = ['']
                count = 0
                for index in range(len(keywords)):
                    best_match = process.extract(keywords[index][0],seen)
                    if best_match[0][1]>80:
                        continue
                    cols[count].button(keywords[index][0])
                    count+=1
                    
                    if count>4:
                        break
                    seen.append(keywords[index][0])
                
                st.subheader('Compare blog')
                blog2 = st.selectbox('select blog for comparison',[None]+[x for x in df['title'].values if x!=blog])
                if blog2:
                        st.markdown('<i><b>Blog 1</b>: {} <br> <b>Blog 2</b>: {}</i>'.format(blog,blog2),unsafe_allow_html=True)
                        text_tokens2,stats2, original_text2 = get_blog_text(df[df['title']==blog2]['url'].values[0])
                        temp2 = {x:[a for a in y.values()][0] for x,y in df[df['title']==blog2][['avg views per day','read_%','views','reads','claps','upvotes','reading Length (minutes)']].to_dict().items()}
                        stats2.update(temp2)
                        temp2 = pd.DataFrame(columns=['stat_2','value_2'])

                        temp2['stat_2'] = [x for x in stats2.keys()]
                        temp2['value_2'] = [str(x) for x in stats2.values() ]

                        temp = pd.merge(temp.set_index('stat'),temp2.set_index('stat_2'),left_index=True,right_index=True)
                        temp['diff'] = temp.apply(lambda x: int(round(float(x['value'])-float(x['value_2']),0)) ,axis=1)
                        temp['comparison'] = ''

                        for index,row in temp.iterrows():
                                temp.at[index,'comparison'] = utils.comparison(row['diff'],'blog 1','blog 2',index)
                        
                        st.dataframe(temp['comparison'])

                        A = config.sent_model.encode(' '.join(text_tokens))
                        B = config.sent_model.encode(' '.join(text_tokens2))

                        
                        cosine = 1 - spatial.distance.cosine(A,B)

                        speedometer(round(cosine,1))
except:
    pass

        

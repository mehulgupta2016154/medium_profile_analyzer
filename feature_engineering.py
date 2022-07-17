
import pandas as pd

import pandas as pd
import re
from spacy.lang.en.stop_words import STOP_WORDS


import pandas as pd

import datetime
from constant import *

from datetime import datetime
import config

def engineering(df,data):
            df['publication'] = df['collection Id'].transform(lambda x:data['payload']['references']['Collection'][x]['name']) 
            df['publication slug'] = df['collection Id'].transform(lambda x:data['payload']['references']['Collection'][x]['slug']) 
            df['image Url'] = df['Image'].transform(lambda x: "https://miro.medium.com/max/600/{}".format(x['id']))
            df['url'] = df.apply(lambda x: "https://medium.com/{}/{}-{}".format(x['publication slug'],x['title slug'],x['post Id']),axis=1)
            df['first Published At'] = pd.to_datetime(df['first Published At'],unit='ms')
            df['firt Published At Bucket'] =  pd.to_datetime(df['first Published At Bucket'])
            df['read_%'] = (100*df['reads'])//df['views']
            df['weekday'] = df['first Published At'].transform(lambda x:x.weekday())
            
            df['days since published'] = df['first Published At'].transform(lambda x:(datetime.now()-x).days )
            df['avg views per day'] = df['views']//df['days since published']
            df['month'] = df['first Published At'].transform(lambda x: x.month)
            df['year'] = df['first Published At'].transform(lambda x: x.year)
            df['title tokens'] = df['title'].transform(lambda x: [a for a in re.sub('[^A-Za-z]+', ' ',x).lower().split(' ') if a not in STOP_WORDS])
        
            df.drop(['title slug','publication slug','post Id','Image'],axis=1,inplace=True)
            config.df = df
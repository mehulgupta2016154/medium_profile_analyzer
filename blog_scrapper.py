import requests
from bs4 import BeautifulSoup as soup
import json
import re
from nltk.corpus import stopwords
import copy

def get_blog_text(url):
    stats = dict()
    page = requests.get(url)
    html = soup(page.content,'html.parser')
    text = ''
    for x in html.find_all('script'):
        if 'window.__APOLLO_STATE__' in x.text:
                data = json.loads(x.text.split("__ = ")[1])
                for y in data.keys():
                    if 'Paragraph' in y:
                        text+=' {}'.format(data[y]['text'])
    original_text = copy.deepcopy(text)
    stats.update({'total words':len(text.split(' '))})
    text = re.sub('[^a-zA-Z]+',' ', text)
    text = [word.strip() for word in text.split(' ') if word not in stopwords.words()]
    text = [x for x in text if len(x)>2]
    stats.update({'stop words %':(100*len(text))//stats['total words']})
    return text, stats, original_text

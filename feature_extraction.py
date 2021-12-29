import pandas as pd
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

def processText(text):
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    #text = text.str.replace('[^\w\s]','')
    stop = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # remove all urls from df
    text = text.apply(lambda x: remove_url(x))
    # remove all html tags from df
    text = text.apply(lambda x: remove_html(x))
    # remove all emojis from df
    text = text.apply(lambda x: remove_emoji(x))
    round1 = lambda x: clean_text_round1(x)
    text = text.apply(round1)
    round2 = lambda x: clean_text_round2(x)
    text = text.apply(round2)

    return text

def count_en(text):
    doc = nlp(text)
    return len(doc.ents)

def process(df):
    # This is where you can do all your processing

    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    df['Prod_label'] = pd.factorize(df['ProductId'])[0]
    df['User_label'] = pd.factorize(df['UserId'])[0]

    df['temp_s'] = df['Score']
    #count upper in Summary
    df['upper'] = df['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    #count upper in text
    df['upper'] = df['Summary'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

    df['Text'] = df.Text.fillna('')
    df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    pol = lambda x: TextBlob(x).sentiment.polarity
    df['polarity'] = df['Text'].apply(pol) 
    df['Text'] = processText(df['Text'].astype(str))
    df['Text']= df['Text'].astype(str) # Make sure about the correct data type

    df['Summary'] = df.Summary.fillna('')
    df['Summary'] = df['Summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    pol = lambda x: TextBlob(x).sentiment.polarity
    df['Summary'] = df['Summary'].apply(pol)
    df['Summary'] = processText(df['Summary'].astype(str))
    df['Summary']= df['Summary'].astype(str) # Make sure about the correct data type
    nlp = en_core_web_sm.load()
    df['entity_count'] =  df['Text'].fillna('').apply(count_en)
    df['intellectual'] = df['entity_count']/df['word_count']

    pos_dict = ['great', 'best', 'love', 'fun']
    neg_dict = ['awful', 'bad', 'waste', 'boring', 'better']
    df['very_neg'] = df['Text'].fillna('').apply(lambda x: len([x for x in x.split() if x in neg_dict]))
    df['very_pos'] = df['Text'].fillna('').apply(lambda x: len([x for x in x.split() if x in pos_dict]))

    return df


# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

#trainingSet= trainingSet.sort_values('Time')

# Process the DataFrame
train_processed = process(trainingSet)

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)


# more code and details about the project can be found here https://github.com/mernaAlghannam/AmazonReviewsDataPrediction.git
# This code extracts features for the review dataset necessary for predicting amazon rating based on reviews
# data cleaning and analysis code used before feature extraction can be found on github link below
# https://github.com/mernaAlghannam/AmazonReviewsDataPrediction/blob/main/2_Amazon_Review_Data_Visulization.ipynb
# model code can be found here https://github.com/mernaAlghannam/AmazonReviewsDataPrediction/blob/main/predict-svc.py

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
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    return len(doc.ents)

def process(df):
    # This is where you can do all your processing


    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    df['Prod_label'] = pd.factorize(df['ProductId'])[0]
    df['User_label'] = pd.factorize(df['UserId'])[0]

    df['temp_s'] = df['Score']
    #count uppercase words in Summary (they indicate extreme anger or happiness)
    df['upper'] = df['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    #count upper in text (they indicate extreme anger or happiness)
    df['upper'] = df['Summary'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

    df['Text'] = df.Text.fillna('')
    df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    pol = lambda x: TextBlob(x).sentiment.polarity
    #feature that indicate the text's possible polarity (each review is marked as either neg or pos)
    df['polarity'] = df['Text'].apply(pol) 
    df['Text'] = processText(df['Text'].astype(str))
    df['Text']= df['Text'].astype(str) 

    df['Summary'] = df.Summary.fillna('')
    df['Summary'] = df['Summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    pol = lambda x: TextBlob(x).sentiment.polarity
    df['Summary'] = df['Summary'].apply(pol)
    df['Summary'] = processText(df['Summary'].astype(str))
    df['Summary']= df['Summary'].astype(str) 
    # Named entity recognition (NER) — sometimes referred to as entity chunking, extraction, or identification —
    # is the task of identifying and categorizing key information (entities) in text. An entity can be any word or 
    # series of words that consistently refers to the same thing. Every detected entity is classified into a predetermined category. 
    # For example, an NER machine learning (ML) 
    # model might detect the word “super.AI” in a text and classify it as a “Company”.
    df['entity_count'] =  df['Text'].fillna('').apply(count_en)
    # my analysis (https://github.com/mernaAlghannam/AmazonReviewsDataPrediction/blob/main/2_Amazon_Review_Data_Visulization.ipynb) indicates
    #  more intellectuaal reviews should be ranked as more relavant for the model
    df['intellectual'] = df['entity_count']/df['word_count']

    # words in reviews that I found in analysis generaly correlates to revies that got 5 start
    pos_dict = ['great', 'best', 'love', 'fun']
    # words in reviews that I found in analysis generaly correlates to revies that got 1 start
    neg_dict = ['awful', 'bad', 'waste', 'boring', 'better']
    # create a feature that extracts all of the terms from text reviews if they are in possitive dictionary
    df['very_neg'] = df['Text'].fillna('').apply(lambda x: len([x for x in x.split() if x in neg_dict]))
    # create a feature that extracts all of the terms from text reviews if they are in negative dictionary
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




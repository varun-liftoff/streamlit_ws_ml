from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import string
import re

from wordcloud import WordCloud
from wordcloud import STOPWORDS

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from nltk.corpus import stopwords
from nltk import ngrams

from collections import Counter


class FileReference:
    def __init__(self, filename):
        self.filename = filename


file_reference = FileReference('https://storage.googleapis.com/streamlit-poc/WS_ML_Models.xlsx')


@st.cache
def load_file():
    df = pd.read_excel(file_reference.filename)
    df['nsfw_level'] = df['porn_level']
    return df


df = load_file()


label_countplot = px.histogram(df, x="label", color="label", title='Sentiment Countplot')
st.plotly_chart(label_countplot, use_container_width=True)


def published_sentiment_analysis_bar_graph():
    published_sentiment = df.loc[:, ['published', 'label']]
    published_sentiment['published'] = published_sentiment.published.dt.to_period('m')
    published_sentiment = published_sentiment.groupby(['published', 'label']).label.count().unstack()
    published_sentiment = published_sentiment.fillna(0)
    published_sentiment = published_sentiment.reset_index()
    published_sentiment = published_sentiment.sort_values('published')
    published_sentiment['published'] = published_sentiment['published'].apply(
        lambda x: x.strftime('%b-%Y'))

    return px.bar(
        data_frame=published_sentiment,
        x="published",
        y=["negative", "neutral", "positive"],
        barmode='group',
        title='Published - Sentiment Analysis',
    )


st.plotly_chart(
    published_sentiment_analysis_bar_graph(),
    use_container_width=True
)


# option = st.selectbox(
#     "Select Sentiment",
#     ("Positive", "Negative", "Neutral"),
# )
# print(option)

# TOP_DOMAIN = []


@st.cache
def sentiment_analysis_bar_graph(column_name, title):
    sentiment = df.loc[:, [column_name, 'label']]

    # if column_name == 'porn_level':
    sentiment[column_name] = sentiment[column_name].astype(str)
        # column_name = 'nsfw_level'

    if column_name == 'source_type':
        sentiment['source_type'] = sentiment['source_type'].apply(lambda x: x.split(',')[1].replace('_',', '))

    sentiment[column_name].fillna('')

    groupby_sentiment = sentiment.groupby(
        [column_name, 'label']).label.count().unstack()

    groupby_sentiment = groupby_sentiment.fillna(0)
    groupby_sentiment = groupby_sentiment.reset_index()
    groupby_sentiment = groupby_sentiment.sort_values(
        'positive', ascending=False)
    groupby_sentiment = groupby_sentiment[groupby_sentiment[column_name] != 'nan']

    if column_name == 'nsfw_level':
        groupby_sentiment = groupby_sentiment[groupby_sentiment[column_name]!='0']
        print(groupby_sentiment)
    return px.bar(
        data_frame=groupby_sentiment.head(10),
        x=column_name,
        y=["negative", "neutral", "positive"],
        barmode='group',
        title=title,
    )


st.plotly_chart(sentiment_analysis_bar_graph(
    'extra_author_attributes.name',
    'Author - Sentiment Analysis'
))
st.plotly_chart(sentiment_analysis_bar_graph(
    'domain_url',
    'Domain - Sentiment Analysis'
))
st.plotly_chart(sentiment_analysis_bar_graph(
    'source_type',
    'Source Type - Sentiment Analysis'
))
st.plotly_chart(sentiment_analysis_bar_graph(
    'extra_source_attributes.world_data.country',
    'Country - Sentiment Analysis'
))

st.plotly_chart(sentiment_analysis_bar_graph(
    'nsfw_level',
    'NSFW - Sentiment Analysis'
))

# st.subheader("Co-relation between NSFW content and sentiment")
# sol = []
# pornLevelIndex = df.porn_level.value_counts().index.sort_values()
# for x in pornLevelIndex:
#     if x==0:
#         continue
#     temp = []
#     tempdf = df[df.porn_level==x].label.value_counts()
#     tempdfIndex= tempdf.index
#     for i in ['positive', 'negative', 'neutral']:
#         temp.append(tempdf[i] if i in tempdfIndex else 0)
#     sol.append(temp)
# bar1 = [x[0] for x in sol]
# bar2 = [x[1] for x in sol]
# bar3 = [x[2] for x in sol]
# fig, ax = plt.subplots()
# x= np.arange(len(pornLevelIndex)-1)
# width=0.5
# rects1 = ax.bar(x - 0.5, bar1, width, label='postive')
# rects2 = ax.bar(x , bar2, width, label='negative')
# rects2 = ax.bar(x + 0.5, bar3, width, label='neutral')
# ax.set_ylabel('Count')
# ax.set_title('Sentiment breakdown by levels of NSFW content')
# ax.set_xlabel('NSFW Level')
# ax.legend()
# st.pyplot(fig)
# domain_sentiment = df.groupby(
#     ['domain_url', 'label']).label.count().unstack()
# domain_sentiment = domain_sentiment.fillna(0)
# domain_sentiment = domain_sentiment.reset_index()
# domain_sentiment = domain_sentiment.sort_values(
#     option.lower(), ascending=False)
# TOP_DOMAIN = domain_sentiment['domain_url'].head(10).tolist()

# option_2 = st.sidebar.multiselect(
#     "Select Domains",
#     TOP_DOMAIN,
# )
# print(option_2)


# @st.cache
# def word_cloud_title_sentiment(sentiment_label):
#     if not option_2:
#         sentiment_title_series = df[df['label'] == sentiment_label]['title']
#     else:
#         sentiment_title_series = df[
#             (df['label'] == 'negative') & (df['domain_url'].isin(option_2))
#         ]['title']
#     sentiment_title_series = sentiment_title_series.apply(lambda x: str(x))
#     sentiment_title_series = sentiment_title_series.apply(
#         lambda x: x.translate(str.maketrans('', '', string.punctuation)))

#     sentiment_title_text = " ".join(i.lower() for i in sentiment_title_series)

#     stopwords = set(STOPWORDS)
#     stopwords.add('new')

#     wordcloud = WordCloud(
#         width=600,
#         height=600,
#         background_color='white',
#         stopwords=stopwords,
#         min_font_size=10
#     ).generate(sentiment_title_text)
#     return wordcloud
#     # plot the WordCloud image
# fig_wordcloud = plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(word_cloud_title_sentiment(option.lower()))
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.title("Title - Sentiment WordCloud")


# st.pyplot(fig_wordcloud)


# option_3 = st.sidebar.selectbox(
#     "Select Word Frequency Distribution",
#     ['Title', 'Content'],
# )
# print(option_3)


# @st.cache
# def word_freq_dist_plot(sentiment_label, freq_dist_text):
#     sentiment_title_series = df[(df['label'] == sentiment_label)][freq_dist_text]
#     sentiment_title_series = sentiment_title_series.apply(lambda x: str(x))
#     sentiment_title_series = sentiment_title_series.apply(
#         lambda x: x.translate(str.maketrans('', '', string.punctuation)))
#     sentence = " ".join(i.lower() for i in sentiment_title_series)

#     tokens = re.findall(r'\w+', sentence)
#     sw = stopwords.words('english')
#     new_tokens = [word for word in tokens if word not in sw]

#     counted = Counter(new_tokens)
#     counted_2 = Counter(ngrams(new_tokens, 2))

#     word_freq = pd.DataFrame(
#         counted.items(),
#         columns=['word', 'frequency']
#     ).sort_values(by='frequency', ascending=False)

#     word_pairs = pd.DataFrame(
#         counted_2.items(),
#         columns=['pairs', 'frequency']
#     ).sort_values(by='frequency', ascending=False)

#     return word_freq, word_pairs

# word_freq, word_pairs = word_freq_dist_plot(option.lower(), option_3.lower())
# fig_freq_dist, axes = plt.subplots(1, 2, figsize=(20, 10))

# plt.title("Word Frequency Distribution")
# plt.rcParams.update({'font.size': 18})
# sns.barplot(
#     ax=axes[0],
#     x='frequency',
#     y='word',
#     data=word_freq.head(30),
#     palette="flare"
# )
# sns.barplot(
#     ax=axes[1],
#     x='frequency',
#     y='pairs',
#     data=word_pairs.head(30),
#     palette="ch:start=.2,rot=-.3"
# )
# fig_freq_dist.tight_layout()


# st.pyplot(fig_freq_dist)

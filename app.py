import spacy
import string
from newsapi import NewsApiClient
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import matplotlib.pyplot as plt


nlp_eng = spacy.load("en_core_web_lg")
newsapi = NewsApiClient(api_key='2901ecaab5014725b173a28916c57e13')


def get_keywords_eng(token):
    result = []
    punctuation = string.punctuation

    for i in token:
        if (i in nlp_eng.Defaults.stop_words or i in punctuation):
            continue
        else:
            result.append(i)
    return result


def main():
    temp = newsapi.get_everything(q='coronavirus', language='en',
                                  from_param='2022-02-27', to='2022-03-27', sort_by='relevancy')

    cleaned_articles = []
    for i, article in enumerate(temp['articles']):
        title = article['title']
        description = article['description']
        content = article['content']
        date = article['publishedAt']
        cleaned_articles.append({'title': title, 'date': date,
                                 'desc': description, 'content': content})
    df = pd.DataFrame(cleaned_articles)
    df = df.dropna()
    df.head()

    tokenizer = RegexpTokenizer(r'\w+')

    results = []
    for content in df.content.values:
        content = tokenizer.tokenize(content)
        results.append([x[0]
                       for x in Counter(get_keywords_eng(content)).most_common(5)])

    df['keywords'] = results

    text = str(results)
    wordcloud = WordCloud(max_font_size=50, max_words=100,
                          background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()

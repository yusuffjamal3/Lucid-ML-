'''
Deployed page summarizer
as a web app and an api endpoint
'''
import re
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import requests
from pandas.io.json import json_normalize
import summarizer
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def home():
    ''' Renders the home page of web app'''
    return render_template('index.html')

@app.route('/url_page_summarize')
def url_page_summarize():
    ''' Renders the page that
        will take in the url for summary
        in the web app'''
    return render_template('url_page_summarize.html')

@app.route('/article_summarize')
def article_summarize():
    ''' Renders the page that
        will take in the article for summary
        in the web app'''
    return render_template('article_summarize.html')

@app.route('/summarized_article', methods=['POST'])
def summarized_article():
    ''' Function that takes in an article
    and generates term frequencies and document frequencies
    and then ranks them as important terms which is used
    as a summary'''
    text_string = [x for x in request.form.values()]
    for texts in text_string:
        text = texts.lower()
    text = re.sub("[^a-zA-Z.,]", " ", text)
    freq_table = summarizer.generate_frequency_table(text)
    sentences = sent_tokenize(text)
    sentence_scores = summarizer.rank_sentences(sentences, freq_table)
    threshold = summarizer.calc_average_rank(sentence_scores)
    summary = summarizer.summarize(sentences, sentence_scores, 1.3 * threshold)
    return render_template('summarized_url.html', summary=summary)

@app.route('/summarized_url', methods=['POST'])
def summarized_url():
    ''' Function that takes in a url and reads the text
    and uses beautifulsoup to parse the html and extract
    text data from the p tags and generates term
    and document frequencies and then ranks them as
    important terms which is used as a summary
    '''
    text_array = []
    url_string = [x for x in request.form.values()]
    for urls in url_string:
        url = urls.lower()
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'lxml')
    for paragraph in soup.find_all('p'):
        paragraph = paragraph.text
        text_array.append(paragraph)
    text = " ".join(text_array)
    text = re.sub("[^a-zA-Z.,]", " ", text)

    freq_table = summarizer.generate_frequency_table(text)
    sentences = sent_tokenize(text)
    sentence_scores = summarizer.rank_sentences(sentences, freq_table)
    threshold = summarizer.calc_average_rank(sentence_scores)
    summary = summarizer.summarize(sentences, sentence_scores, 1.3 * threshold)
    return render_template('summarized_url.html', summary=summary)


@app.route('/summarized_article_api', methods=['POST'])
def summarized_article_api():
    ''' Function that takes in an article as a json object
    and generates term frequencies and document frequencies
    and then ranks them as important terms which is used
    as a summary.
    This function is  the api endpoint which will be called
    and it takes in data in the format {'text': 'the full text'}
    '''
    json_data = request.get_json(force=True)
    json_df = json_normalize(json_data)
    for texts in json_df.text:
        text = texts.lower()
    text = re.sub("[^a-zA-Z0-9.,]", " ", text)

    freq_table = summarizer.generate_frequency_table(text)
    sentences = sent_tokenize(text)
    sentence_scores = summarizer.rank_sentences(sentences, freq_table)
    threshold = summarizer.calc_average_rank(sentence_scores)
    summary = summarizer.summarize(sentences, sentence_scores, 1.3 * threshold)
    return jsonify(summary)


@app.route('/summarized_url_api', methods=['POST'])
def summarized_url_api():
    ''' Function that takes in a url as a json object
    and uses beautifulsoup to parse the html and extract
    text data from the p tags and generates term
    frequencies and document frequencies and then ranks
    them as important terms which is used as a summary.
    This function is  the api endpoint which will be called
    and it takes in data in the format {'url': 'the url '}
    '''
    text_array = []
    json_data = request.get_json(force=True)
    json_df = json_normalize(json_data)
    for urls in json_df.url:
        url = urls.lower()
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'lxml')
    for paragraph in soup.find_all('p'):
        paragraph = paragraph.text
        text_array.append(paragraph)
    text = " ".join(text_array)
    text = re.sub("[^a-zA-Z0-9.,]", " ", text)

    freq_table = summarizer.generate_frequency_table(text)
    sentences = sent_tokenize(text)
    sentence_scores = summarizer.rank_sentences(sentences, freq_table)
    threshold = summarizer.calc_average_rank(sentence_scores)
    summary = summarizer.summarize(sentences, sentence_scores, 1.3 * threshold)
    return jsonify(summary)

if __name__ == "__main__":
    app.run(debug=True)

<div align="center">

![hng](https://res.cloudinary.com/iambeejayayo/image/upload/v1554240066/brand-logo.png)

<br>

</div>

# Lucid - Ml - Page Summarizer

## INTRODUCTION
This is the Page Summarizer built by the Lucid-ML Team. The ML Model summarizes an article/page based on the URL of the page specified. 

## REQUIRED LIBARIES
```bash
- Regex
- Flask
- BeautifulSoup
- NLTK
- Requests
- Pandas
```

## Summarizer Script
- Imported Relevant Modules
```bash
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
```

- Generated A Frequency Table
  - Tokenized the text into Words
  - Stemmed each of the tokenized words
  - Calculated the number of times they appear
  
- Ranked sentences based on the total frequency of words in them.

- Calculated the average value of a sentence from original text 

- Summarized the text by taking the most frequent words

## ML Model
- Imported modules and the summarizer script
```bash
import re
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import requests
from pandas.io.json import json_normalize
import summarizer
```

- Used Flask to set url paths

- Rendered two pages (The first page takes text, the second takes urls)

- Used beautiful soup to parse the html and extract text data from the p tags
  
- Used the functions created within the summarizer script to summarize text entered.

- Created an API endpoint which generates them in the format {'text': 'the summary'} or {'url': 'the summary'}

## HOW TO SETUP THE MODEL
- Download Docker for your respective OS.

- Clone the Repo
```bash
git clone https://github.com/pibeebeks/Lucid-ML-.git
```

- Enter the Directory
```bash
cd Lucid-ML-
```

- Run Docker compose to build the Image and start the container:
```bash
docker-compose up
```

- Open your browser and type in this URL on proceedures for using the model
```bash
http://127.0.0.1:5000/
```

- Stop the application by running the below command from within your project directory in the second terminal, or by hitting CTRL+C in the original terminal where you started the app:
```bash
docker-compose down
```



import json
import string
from nltk.stem import PorterStemmer

def keyword_search(query):
    # Load movie data from JSON storage
    with open('data/movies.json', 'r') as f:
        data = json.load(f)

    stemmer = PorterStemmer()
    movies_list = data["movies"]
    results = []
    i=1

    # Preprocess the user query into stemmed roots
    query_tokens = [stemmer.stem(token) for token in remove_stopwords(tokenize(query))]

    for item in movies_list:
        # Preprocess each title to match the query's format
        title_tokens = [stemmer.stem(token) for token in remove_stopwords(tokenize(item['title']))]

        # Check if any stemmed query word exists within the stemmed movie title tokens
        if any(q in t for q in query_tokens for t in title_tokens):
            results.append(item)
            print(f"{i}. {item['title']}")
            i+=1

        # Limit retults
        if len(results) == 5:
            break

    return results

def tokenize(text):
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return cleaned_text.split()

def remove_stopwords(text):
    with open("data/stopwords.txt") as f:
        words = f.read()

    words = words.splitlines()
    return [word for word in text if word not in words]
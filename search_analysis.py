#imports
import requests #send HTTP requests to web pages and retrieve their content
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

import ssl
nltk.download('punkt')

search_query = input("Enter Query: ")
search_engine_url = "https://www.google.com/search?q="
 
# Set headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# retrieve the search result page and extract information 
response = requests.get(search_engine_url + search_query, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

search_results = []
for result in soup.find_all("div", class_="tF2Cxc"):
    title = result.find("h3").text
    url = result.find("a")["href"]
    
    snippet_element = result.find("span", class_="aCOpRe")
    if snippet_element:
        snippet = snippet_element.text
    else:
        snippet = ""
    
    search_results.append({"title": title, "url": url, "snippet": snippet})

# pre-process the search results
stemmer = PorterStemmer()
preprocessed_results = []

for result in search_results:
    title_tokens = word_tokenize(result["title"])
    snippet_tokens = word_tokenize(result["snippet"])

    title_tokens = [stemmer.stem(token) for token in title_tokens]
    snippet_tokens = [stemmer.stem(token) for token in snippet_tokens]

    preprocesses_title = " ".join(title_tokens)
    preprocesses_snippet = " ".join(snippet_tokens)

    preprocessed_results.append({"title": preprocesses_title, "snippet": preprocesses_snippet})

#print out the search results
for result in search_results:
    print("Title:", result["title"])
    print("Snippet:", result["snippet"])
    if "url" in result:
        print("URL:", result["url"])
    print("-" * 50)

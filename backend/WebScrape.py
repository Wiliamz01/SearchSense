#imports
import requests #send HTTP requests to web pages and retrieve their content
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from urllib.parse import urlparse, urljoin
import pandas as pd
from nltk.corpus import stopwords
import http.client
import re
import csv


import ssl
nltk.download('punkt')
#remove filler words from an example column
nltk.download("stopwords")
nltk.download('wordnet')



import concurrent.futures
from tqdm import tqdm
#Working implementation

def get_df():

        session = requests.Session()
        retry = Retry(connect=2, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        # Define the search query and retrieve search results
        search_query = input("What would you like to search up?")
        search_engine_url = "https://www.google.com/search?q="
        
        # Set headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Number of search result pages to retrieve (each page typically has 10 results)
        num_pages = 5
        
        all_urls = []
        search_results = []
        
        for page in range(num_pages):
            # Calculate the start index for the current page
            start_index = page * 10
        
            try:
                # Make a GET request to retrieve the search result page
                response = session.get(search_engine_url + search_query + f"&start={start_index}", headers=headers)
                response.raise_for_status()  # Raise an exception for non-2xx status codes
        
            except requests.exceptions.RequestException as e:
                print("Error occurred for page", page+1)
                print("Exception:", e)
                continue
        
            # Parse and extract information from the search result page
            soup = BeautifulSoup(response.content, "html.parser")
        
            for result in soup.select("div.g"):
                title_element = result.select_one("h3")
                if title_element:
                    title = title_element.get_text()
                    url_element = result.find("a")
                    if url_element:
                        url = url_element["href"]
                        parsed_url = urlparse(url)
                        if parsed_url.scheme == "":
                            # Relative URL, convert it to absolute URL
                            base_url = response.url
                            url = urljoin(base_url, url)
                        all_urls.append(url)
        
        
        def get_response(url):
            try:
                response = session.get(url, verify=True, headers=headers)
                result_soup = BeautifulSoup(response.content, "html.parser")
                text = result_soup.get_text()
                text = ' '.join(text.split())
                search_results.append({"Title": title, "URL": url, "Text": text})
            except http.client.IncompleteRead:
                print("Error: Incomplete read for:", url)
                return None
            except Exception as e:
                print("Error: Failed to retrieve content for:", url)
                print("Exception:", e)
                return None
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
        for cur_url in all_urls:
            futures = [
                executor.submit(
                    get_response,
                    cur_url
                )
            ]
        
        results = []
        with tqdm(total=len(futures), desc="Scraping...") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(results)
                pbar.update(1)
        
            
        print("Grabbed search results")
        
        # Process the search results as desired
        
        pattern = r"\b(?:skip|main|content|table|menu|displayclass|pagesubpageproperty|ad|copyright|com|ssc|f|wiki|courses|course|sign|up|log|facebook|google|email|forgot|password|user|existing|manually|already|account|f|frac|cdots|hspace|mm|ikamusumefan|gif|org|https|loading|align|p|x|c|extensionprocessorqueryprovider|libretexts|pageindex|dfrac|mathrm|newcommand|norm|extensionprocessorqueryprovider|pagesubpageproperty)\b"
        
        df = pd.DataFrame(search_results)
        df_bart = pd.DataFrame(search_results)
        df_bart_unlem = pd.DataFrame(search_results)
        # Extract domain names from URLs and set as the index for the dataframe
        tld_pattern = re.compile(r'\.[a-zA-Z]+$')
        df['Domain'] = df['URL'].apply(lambda url: re.sub(tld_pattern, '', urlparse(url).netloc.replace("www.", "")))
        df.insert(0, 'Domain', df.pop('Domain'))
        
        df_bart['Domain'] = df_bart['URL'].apply(lambda url: re.sub(tld_pattern, '', urlparse(url).netloc.replace("www.", "")))
        df_bart.insert(0, 'Domain', df_bart.pop('Domain'))
        #Handles the "Some characters could not be decoded, and were replaced with REPLACEMENT CHARACTER" exception
        df_bart['Text'] = df_bart['Text'].str.replace('\ufffd', '')
        
        df_bart_unlem['Domain'] = df_bart_unlem['URL'].apply(lambda url: re.sub(tld_pattern, '', urlparse(url).netloc.replace("www.", "")))
        df_bart_unlem.insert(0, 'Domain', df_bart_unlem.pop('Domain'))
        #Handles the "Some characters could not be decoded, and were replaced with REPLACEMENT CHARACTER" exception
        df_bart_unlem['Text'] = df_bart_unlem['Text'].str.replace('\ufffd', '')
        
        #unfiltered dataframe
        print(len(df))
        df.head()
        
        
        
        stop_words = stopwords.words("english") #english filler words
        
        def preprocess(text):
            '''
            Input a text block and filter out unneeded characters
            returns a filtered text block in the form of a str
            Function filters whitespace, numbers, special characters, stopwords; handles case normalization 
            '''
            # Remove special characters and numbers
            cleaned_text = re.sub(r"[^a-zA-Z]+", " ", text)
            
            # Convert to lowercase and split into words
            words = cleaned_text.lower().split()
            
            # Remove stop words and single-character words
            filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
            lemmatized_text = " ".join([WordNetLemmatizer().lemmatize(word) for word in filtered_words])
            
            return lemmatized_text
        
        def bart_preprocess(text):
        
            cleaned_text = re.sub(pattern,"",text, flags=re.IGNORECASE)
        
            return cleaned_text
        
        
        print(len(df_bart["Text"][0]))
        #Properly using the apply() function onto a pandas column
        #removes both filler words and special characters
        # df["Text"] = df["Text"].apply(bart_preprocess)
        df_bart["Text"] = df_bart["Text"].apply(bart_preprocess)

        df_bart.to_csv('bart.csv')

        return df_bart



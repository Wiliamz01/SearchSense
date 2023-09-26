'''

do the web scrape
after getting that, add two lines where query = input
ask user


get the dataframe

deploy the model
run through the model

generate scores for each entry
create a new column and apply function to new column

sort the data frame in any particular order

display all the search results based on the dataframe

'''


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


nltk.download('punkt')
#remove filler words from an example column
nltk.download("stopwords")
nltk.download('wordnet')


import concurrent.futures
from tqdm import tqdm
#Working implementation

#model imports
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first available GPU (index 0)
else:
    device = torch.device("cpu")  # Use CPU if no GPU is available

print(f"Using device: {device}")

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
#Load the model
model = DistilBertForSequenceClassification.from_pretrained("C:/Users/William Zhang/OneDrive/Desktop/Search_Sense_Model").to(device)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def get_result():    
    #Remove <a> tags from search results 7/19/23
    session = requests.Session()
    adapter = HTTPAdapter()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    # Define the search query and retrieve search results
    search_query = input("Enter your search query: ")
    search_engine_url = "https://www.google.com/search?q="

    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Number of search result pages to retrieve (each page typically has 10 results)
    num_pages = 5

    all_urls = []
    search_results = []
    all_titles = []

    for page in range(num_pages):
        # Calculate the start index for the current page
        start_index = page * 10

        try:
            # Make a GET request to retrieve the search result page
            response = session.get(search_engine_url + search_query + f"&start={start_index}", headers=headers)
            response.raise_for_status()  # Raise an exception for non-2xx status codes

        except requests.exceptions.RequestException as e:
            # print("Error occurred for page", page+1)
            # print("Exception:", e)
            continue

        # Parse and extract information from the search result page
        soup = BeautifulSoup(response.content, "html.parser")

        for result in soup.select("div.g"):
            title_element = result.select_one("h3")
            if title_element:
                title = title_element.get_text()
                all_titles.append(title)
                url_element = result.find("a")
                if url_element:
                    url = url_element["href"]
                    parsed_url = urlparse(url)
                    if parsed_url.scheme == "":
                        # Relative URL, convert it to absolute URL
                        base_url = response.url
                        url = urljoin(base_url, url)
                    all_urls.append(url)

    def get_response(url, title):
        try:
            response = session.get(url, verify=True, headers=headers)
            result_soup = BeautifulSoup(response.content, "html.parser")


            # Remove <a> tags and their content from the parsed HTML content
            for a_tag in result_soup.find_all("a"):
                # Check if a <b> tag exists before or after the <a> tag
                if a_tag.find_previous_sibling("b") or a_tag.find_next_sibling("b"):
                    continue  # Keep the <a> tag and its content
                else:
                    a_tag.decompose()  # Remove the <a> tag and its content


            # Get the clean text without <a> tags
            text = result_soup.get_text()
            text = ' '.join(text.split())
            search_results.append({"Title": title, "URL": url, "Text": text})
        except http.client.IncompleteRead:
            # print("Error: Incomplete read for:", url)
            return None
        except Exception as e:
            # print("Error: Failed to retrieve content for:", url)
            # print("Exception:", e)
            return None

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    for cur_url, cur_title in zip(all_urls, all_titles):  
        futures = [
            executor.submit(
                get_response,
                cur_url,
                cur_title
            )
        ]


    results = []
    with tqdm(total=len(futures), desc="Scraping...") as pbar:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(results)
            pbar.update(1)


    print("Search results retrieved successfully")


    # Process the search results as desired        
    df = pd.DataFrame(search_results)
    # Extract domain names from URLs and set as the index for the dataframe
    tld_pattern = re.compile(r'\.[a-zA-Z]+$')
    df['Domain'] = df['URL'].apply(lambda url: re.sub(tld_pattern, '', urlparse(url).netloc.replace("www.", "")))
    df.insert(0, 'Domain', df.pop('Domain'))


    #Carlos implementation, removing beginning and end instances of text without "buzzword 7/14"
    def preprocess(text):
        '''
        Input a text block and filter out unneeded characters
        Returns a filtered text block in the form of a string
        Function keeps punctuation, capitalization, stopwords, and special characters;
        removes LaTeX, keeps plain English, and strips beginning and end text
        '''
        # Remove LaTeX expressions
        cleaned_text = re.sub(r"\$.*?\$", "", text)

        stop_words = set(stopwords.words("english"))
        query_words = search_query.split()
        non_stop_word = next((word for word in query_words if word.lower() not in stop_words), None)

        # Strip beginning text
        text_words = cleaned_text.split()

        count = 0
        start_index = 0
        for i, word in enumerate(text_words):
            if word.lower() == non_stop_word.lower():
                count += 1
                if count == 4:
                    start_index = i
                    break

        # Strip ending text
        count = 0
        end_index = len(text_words)
        for i in range(len(text_words) - 1, -1, -1):
            if text_words[i].lower() == non_stop_word.lower():
                count += 1
                if count == 2:
                    end_index = i + 1
                    break

        # Hardcoding common navigation words in scraped text
        pattern = r"\b(?:skip|main|content|table|menu|displayclass|pagesubpageproperty|ad|copyright|com|ssc|f|wiki|courses|course|sign|up|log|facebook|google|email|forgot|password|user|existing|manually|already|account|f|frac|cdots|hspace|mm|ikamusumefan|gif|org|https|loading|align|p|x|c|extensionprocessorqueryprovider|libretexts|pageindex|dfrac|mathrm|newcommand|norm|extensionprocessorqueryprovider|pagesubpageproperty|browser|firefox|install|chrome|edge|wikimedia|commonswikibookswikiquotewikiversity|t)\b"
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = " ".join(text_words[start_index:end_index])


        return cleaned_text


    df["Text"] = df["Text"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    df["Text"] = df["Text"].apply(preprocess)

    #remove yt videos
    df = df[~df['Domain'].str.contains('youtube')]
    #remove null or empty values
    df.dropna(subset=["Text"], inplace=True)
    #remove entries with less than 25 words
    df = df[df["Text"].apply(lambda x: len(x.split()) >= 25)]

    def get_parts_of_text(content):
        words = content.split()  

        # First 50 words
        x = " ".join(words[:50])

        # Middle 300 words
        middle_index = len(words) // 2  
        y_start = max(0, middle_index - 150)  
        y_end = middle_index + 150
        y = " ".join(words[y_start:y_end])

        # Last 50 words
        z = " ".join(words[-50:])

        return x, y, z


    # Define a function to make predictions
    def make_prediction(text, model, tokenizer):

        # Tokenize the text
        content_len = len(text.split())
        first_50_words, middle_300_words, last_50_words = get_parts_of_text(text)

        text_final_input = f"""
        Total Content Length:{content_len}

        Text Snippets:
        First 50 words of content:
        {first_50_words}
        Middle 300 words of content:
        {middle_300_words}
        Last 50 words of content:
        {last_50_words}
        """


        encoding = tokenizer.encode_plus(
            text_final_input,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        # Set model to evaluation mode and make prediction
        model.eval()
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]

        # Convert output to label
        predicted_score = output.item()
        return predicted_score
    
    df['predicted_scores'] = df['Text'].apply(lambda x: make_prediction(x, model, tokenizer))
    df = df.sort_values(by = ["predicted_scores"])
    
    for index, row in df.iterrows():
        title = row['Title']
        url = row['URL']
        predicted_score = row['predicted_scores']

        if predicted_score < 0.5:
            difficulty = "Low"
        elif 0.5 <= predicted_score <= 0.88:
            difficulty = "Moderate"
        else:
            difficulty = "High"

        print(f"""
        Title: {title}
        URL: {url}
        Difficulty: {difficulty} Model Score: ({predicted_score})
        --------------------------------------------------------
        """)
    
    return df

import re
import json
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

try:
    stopword_list = set(stopwords.words('english'))
except Exception:
    import nltk
    nltk.download('stopwords')
    stopword_list = set(stopwords.words('english'))

negation_words = {'not', 'cannot', 'no', 'nt', 'but', 'however', 'although', 'except'}
stopword_list = stopword_list - negation_words

slang_file_path = os.path.join(os.path.dirname(__file__), 'slang.txt')
try:
    with open(slang_file_path, 'r', encoding='utf-8') as f:
        slangwords = json.load(f)
except FileNotFoundError:
    print(f"Warning: File {slang_file_path} tidak ditemukan, slang replacement dilewati.")
    slangwords = {}

custom_phrases = {
    "not good": "not_good", "not bad": "not_bad",
    "no problem": "no_problem", "don't like": "dont_like", "not working": "not_working"
}

def preprocess_text(text):
    text = str(text).lower()
    
    # 1. Regex Cleaning (Hanya huruf dan spasi)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # Normalisasi huruf berulang
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Custom Phrase
    for phrase, replacement in custom_phrases.items():
        text = text.replace(phrase, replacement)
        
    # 3. Tokenize
    tokens = text.split()
    
    # 4. Slang replacement & Stopwords & Stemming
    clean_tokens = [
        stemmer.stem(slangwords.get(token, token)) 
        for token in tokens 
        if slangwords.get(token, token) not in stopword_list
    ]
    
    return " ".join(clean_tokens)

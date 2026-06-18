import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

def main():
    # Make sure we have the required NLTK data
    nltk.download('stopwords')

    print("Loading data...")
    df = pd.read_csv('data/hasil_pengumpulan_ulasan_deepseek_eng_8_mei.csv')
    df = df[['score', 'content']].dropna(subset=['content']).drop_duplicates()
    
    print("Preprocessing text...")
    stemmer = SnowballStemmer("english")
    stopword_list = set(stopwords.words('english'))
    negation_words = {'not', 'cannot', 'no', 'nt', 'but', 'however', 'although', 'except'}
    stopword_list = stopword_list - negation_words
    
    try:
        with open('backend/preprocess/slang.txt', 'r', encoding='utf-8') as f:
            slangwords = json.load(f)
    except FileNotFoundError:
        print("Warning: slang.txt not found, proceeding without it.")
        slangwords = {}
        
    custom_phrases = {
        "not good": "not_good", "not bad": "not_bad",
        "no problem": "no_problem", "don't like": "dont_like", "not working": "not_working"
    }

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        for phrase, replacement in custom_phrases.items():
            text = text.replace(phrase, replacement)
            
        tokens = text.split()
        clean_tokens = [
            stemmer.stem(slangwords.get(token, token)) 
            for token in tokens 
            if slangwords.get(token, token) not in stopword_list
        ]
        return " ".join(clean_tokens)

    df['final_clean_text'] = df['content'].apply(preprocess_text)
    df = df[df['final_clean_text'] != ''].dropna(subset=['final_clean_text']).reset_index(drop=True)
    
    print("Labeling with VADER...")
    analyzer = SentimentIntensityAnalyzer()
    def analyze_sentiment_english(text):
        score = analyzer.polarity_scores(str(text))
        compound = score['compound']
        if compound >= 0.05:
            return 'positif'
        elif compound <= -0.05:
            return 'negatif'
        else:
            return 'netral'
            
    df['sentiment'] = df['final_clean_text'].apply(analyze_sentiment_english)
    df = df[df['sentiment'] != 'netral'].reset_index(drop=True)
    
    print("Undersampling to balance classes...")
    min_class_size = df['sentiment'].value_counts().min()
    df_positif = df[df['sentiment'] == 'positif'].sample(n=min_class_size, random_state=42)
    df_negatif = df[df['sentiment'] == 'negatif'].sample(n=min_class_size, random_state=42)
    df = pd.concat([df_positif, df_negatif]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Vectorizing...")
    X = df['final_clean_text']
    y = df['sentiment']
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    print("Training SVM...")
    best_svm = SVC(kernel='linear', class_weight='balanced', C=1)
    best_svm.fit(X_tfidf, y)
    
    print("Saving model and vectorizer...")
    joblib.dump(best_svm, 'backend/model/model.pkl')
    joblib.dump(tfidf_vectorizer, 'backend/model/vectorizer.pkl')
    
    print("Done!")

if __name__ == '__main__':
    main()

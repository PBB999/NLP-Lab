import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd  # For table formatting

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""

# Tokenize into sentences
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

corpus = []
for sent in sentences:
    sent = re.sub('[^a-zA-Z]', ' ', sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    corpus.append(sent)

print("\n Cleaned Corpus:")
for idx, text in enumerate(corpus, start=1):
    print(f"  {idx}. {text}")

# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
independentFeatures = cv.fit_transform(corpus).toarray()

df_bow = pd.DataFrame(independentFeatures, columns=cv.get_feature_names_out())
print("\n Bag of Words (CountVectorizer):")
print(df_bow.to_markdown(index=False))

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()

df_tfidf = pd.DataFrame(independentFeatures_tfIDF, columns=tfidf.get_feature_names_out())
print("\n TF–IDF Vectorizer:")
print(df_tfidf.round(3).to_markdown(index=False))

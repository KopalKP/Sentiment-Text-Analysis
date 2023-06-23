import string
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the text data from a CSV file
data = pd.read_csv('2016_US_election_tweets_100k.csv')
text_column = 'tweet_text'  # Replace 'tweet_text' with the actual column name containing the text data

# Clean the text by removing punctuation and stopwords
cleaned_text = []
stop_words = set(stopwords.words('english'))
for tweet in data[text_column]:
    if isinstance(tweet, str):
        lower_case = tweet.lower()
        cleaned = lower_case.translate(str.maketrans('', '', string.punctuation))
        tokenized_words = word_tokenize(cleaned)
        final_words = [word for word in tokenized_words if word not in stop_words]
        cleaned_text.append(' '.join(final_words))
    else:
        cleaned_text.append('')

# Initialize and fit the vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_text)
y = [1] * len(cleaned_text)  # Assuming all tweets have positive sentiment

# Initialize and fit the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Load emotion list from file
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", "").replace(",", " ").replace("'", "").strip()
        word, emotion = clear_line.split(':')

        for tweet in cleaned_text:
            if word in tweet:
                emotion_list.append(emotion)

# Count the occurrences of each emotion
w = Counter(emotion_list)
print(w)

# Perform sentiment analysis using Vader
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    compound_score = score['compound']
    if compound_score > 0:
        sentiment = "Positive"
    elif compound_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, compound_score

# Perform sentiment analysis using Multinomial Naive Bayes
def naive_bayes_sentiment_analyse(sentiment_text):
    predicted_sentiment = clf.predict(vectorizer.transform([sentiment_text]))[0]
    if predicted_sentiment == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    return sentiment

# Perform sentiment analysis using Vader
sentiment_vader, score_vader = sentiment_analyse(cleaned_text[0])
print(f"Sentiment (Vader): {sentiment_vader}")
print(f"Score (Vader): {score_vader}")

# Perform sentiment analysis using Multinomial Naive Bayes
sentiment_naive_bayes = naive_bayes_sentiment_analyse(cleaned_text[0])
print(f"Sentiment (Naive Bayes): {sentiment_naive_bayes}")

# Evaluate the Multinomial Naive Bayes model
X_eval = vectorizer.transform(cleaned_text)
y_eval = [1] * len(cleaned_text)  # Assuming all tweets have positive sentiment
y_pred = clf.predict(X_eval)
print("Classification Report (Naive Bayes):")
print(classification_report(y_eval, y_pred))

# Plot the emotion distribution
fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()

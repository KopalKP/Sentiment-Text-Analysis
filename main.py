import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the text data
text = open('text.txt', encoding='utf-8').read()
lower_case = text.lower()

# Clean the text by removing punctuation
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
print(cleaned_text)

# Tokenize the cleaned text
tokenized_words = word_tokenize(cleaned_text)
print(tokenized_words)

# Remove stopwords from the tokenized words
stop_words = set(stopwords.words('english'))
final_words = [word for word in tokenized_words if word not in stop_words]
print(final_words)

# Initialize and fit the vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(final_words)
y = [1] * len(final_words)  # Assuming all words have positive sentiment

# Initialize and fit the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Load emotion list from file
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", "").replace(",", " ").replace("'", "").strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
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
sentiment_vader, score_vader = sentiment_analyse(cleaned_text)
print(f"Sentiment (Vader): {sentiment_vader}")
print(f"Score (Vader): {score_vader}")

# Perform sentiment analysis using Multinomial Naive Bayes
sentiment_naive_bayes = naive_bayes_sentiment_analyse(cleaned_text)
print(f"Sentiment (Naive Bayes): {sentiment_naive_bayes}")

# Evaluate the Multinomial Naive Bayes model
X_eval = vectorizer.transform(final_words)
y_eval = [1] * len(final_words)  # Assuming all words have positive sentiment
y_pred = clf.predict(X_eval)
print("Classification Report (Naive Bayes):")
print(classification_report(y_eval, y_pred))

# Plot the emotion distribution
fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()

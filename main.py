import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = open('text.txt', encoding='utf-8').read()
lower_case = text.lower()

cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
print(cleaned_text)

tokenized_words = word_tokenize(cleaned_text)
print(tokenized_words)

final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)
print(final_words)

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", "").replace(",", " ").replace("'", "").strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)

print(emotion_list)

w = Counter(emotion_list)
print(w)

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

sentiment, score = sentiment_analyse(cleaned_text)
print(f"Sentiment: {sentiment}")
print(f"Score: {score}")

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()

import GetOldTweets3 as got
import string

def get_tweets():
    # Set tweet criteria using the GetOldTweets3 library
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(('donald trump')) \
        .setSince("2018-01-01") \
        .setUntil("2019-02-28") \
        .setMaxTweets(1000)

    # Retrieve tweets based on the specified criteria
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    # Store the text of each tweet in a list
    text_tweets = [[tweet.text] for tweet in tweets]
    return text_tweets

# Initialize an empty string to store the concatenated tweets
text = ""

# Get the tweets and concatenate their text
text_tweets = get_tweets()
length = len(text_tweets)

for i in range(0, length):
    text = text_tweets[i][0] + " " + text

# reading text file
# text = open("read.txt", encoding="utf-8").read()
# Convert the text to lowercase
lower_case = text.lower()

# Remove punctuations from the text
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Split the cleaned text into individual words
tokenized_words = cleaned_text.split()

# Define a list of stopwords to be removed
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

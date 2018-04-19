import csv

from nltk import word_tokenize
from nltk.sentiment.util import mark_negation

#splitNum is used to take a subsample of our dataset
def loadDataset(fileName, splitNum, tweetColumn=5):
    with open(fileName, 'rU') as trainingInput:
        # detect the "dialect" of this type of csv file
        try:
            dialect = csv.Sniffer().sniff(trainingInput.read(1024))
        except:
            # if we fail to detect the dialect, defautl to Microsoft Excel
            dialect = 'excel'
        trainingInput.seek(0)
        trainingRows = csv.reader(trainingInput, dialect)

        rowCount = 0

        allTweets = []
        allTweetSentiments = []
        allRows = []
        for row in trainingRows:
            rowCount += 1
            if rowCount % splitNum == 0:
                # csv only gives us an iterable, not the data itself
                allTweets.append(row[tweetColumn])
                allTweetSentiments.append(row[0])
                allRows.append(row)

    return allTweets, allTweetSentiments, allRows

def normalizeClasses(arr):
    normalized = []
    for score in arr:
        if score in ['4', '2']:
            normalized.append(1)
        else:
            normalized.append(0)
    return normalized

def removeBr(text):
    return text.replace("<br />", " ")

def negation(text):
	return mark_negation(word_tokenize(text))
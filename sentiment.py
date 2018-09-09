# -*- coding: utf-8 -*-

import random
import util

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# from sklearn.base import TransformerMixin

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, LinearSVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from nltk.corpus import twitter_samples, stopwords

# class DenseTransformer(TransformerMixin):

#     def transform(self, X, y=None, **fit_params):
#         return X.todense()

#     def fit_transform(self, X, y=None, **fit_params):
#         self.fit(X, y, **fit_params)
#         return self.transform(X)

#     def fit(self, X, y=None, **fit_params):
#         return self

print 'Reading Data...'
# load the "training" data
trainingTweets, trainingSentiment, allRows = util.loadDataset('data/trainingdata.csv', 10)
# load the test data
testTweets, testSentiment, testRows = util.loadDataset('data/testdata.csv', 1)

print 'Parsing class...'
trainingSentiment = util.normalizeClasses(trainingSentiment)
testSentiment = util.normalizeClasses(testSentiment)

print 'Adding emoticons data...'
#adding nltk twits too
positiveTweets = twitter_samples.strings('positive_tweets.json')
negativeTweets = twitter_samples.strings('negative_tweets.json')
positiveSentiment = [1 for x in positiveTweets]
negativeSentiment = [0 for x in positiveTweets]
allTweets = positiveTweets + negativeTweets + trainingTweets
allSentiment = positiveSentiment + negativeSentiment + trainingSentiment

print 'Preparing data...'
sentiment_data = zip(allTweets, allSentiment)
random.shuffle(sentiment_data)
random.shuffle(sentiment_data)
random.shuffle(sentiment_data)
train_X, train_y = zip(*sentiment_data)
test_X = testTweets
test_y = testSentiment

vectorizer = CountVectorizer(
    analyzer = "word",
    ngram_range = (1, 3),
    stop_words = stopwords.words('english'),
    #tokenizer = word_tokenize
    tokenizer = util.negation,
    preprocessor = util.removeBr
 )

classifier = BernoulliNB()
# classifier = KNeighborsClassifier(3)
# classifier = LinearSVC()
# classifier = BernoulliNB()
# classifier = SVC(kernel="linear", C=0.025)
# classifier = SVC(gamma=2, C=1)
# classifier = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
# classifier = DecisionTreeClassifier(max_depth=5)
# classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
# classifier = MLPClassifier(alpha=1)
# classifier = AdaBoostClassifier()
# classifier = GaussianNB()
# classifier = QuadraticDiscriminantAnalysis()

print 'Creating Pipeline...'
model = Pipeline([
	('vectorizer', vectorizer),
	# ('to_dense', DenseTransformer()),
    ('classifier', classifier)
])

print 'Training...'
model.fit(train_X, train_y)
print 'Score:'
print model.score(test_X, test_y)

print 'Saving model...'
joblib.dump(model, 'model.pkl')

# print 'Loading model...'
# model = joblib.load('model.pkl')

print 'Simple test...'
test = ["I feel like :(", "Nice =)", "I don't like it", "Sad =(", "Love it!", "Awesome!!!!", "Nothing really"];
print test
for d in model.predict_proba(test):
	print ":(" if d[0] > d[1] else ":)",
	print "Confidence: %.2f" % round(d[0], 2) if d[0] > d[1] else "Confidence: %.2f" % round(d[1], 2)

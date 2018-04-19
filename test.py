from sklearn.externals import joblib

print 'Loading model...'
model = joblib.load('model.pkl')

print 'Simple test...'
test = ["I feel like mehh...", "Nice =)", "I don't like it", "Sad =(", "I hate you!", "Awesome!!!!", "Nothing really"];
print test
for d in model.predict_proba(test):
  print ":(" if d[0] > d[1] else ":)",
  print "Confidence: %.2f" % round(d[0], 2) if d[0] > d[1] else "Confidence: %.2f" % round(d[1], 2)

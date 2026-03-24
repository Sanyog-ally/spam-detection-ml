import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# load dataset
data = pd.read_csv("dataset.csv")

# separate input and output
X = data["text"]
y = data["label"]

# convert text into numbers
cv = CountVectorizer()
X = cv.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = MultinomialNB()
model.fit(X_train, y_train)

# testing accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# test custom message
msg = ["Free money offer just click now"]
msg = cv.transform(msg)
prediction = model.predict(msg)

print("Prediction:", prediction[0])

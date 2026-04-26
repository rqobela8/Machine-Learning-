import random
from collections import Counter
from sklearn.model_selection import train_test_split
from process_email import build_messages
from model import NaiveBayesClassifier

#Retrieve and split data
data = build_messages()
train_msg,test_msg = train_test_split(
    data,
    train_size=0.75,
    random_state=42)

#create and train data
model = NaiveBayesClassifier(k=0.5)
model.train(train_msg)

#Make Predictions
predictions = [(message, model.predict(message.text))
for message in test_msg]


#Assume that spam_probability > 0.6 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
for message, spam_probability in predictions)

print(confusion_matrix)


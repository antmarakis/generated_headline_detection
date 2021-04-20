import pandas as pd

train = pd.read_csv('defense.csv')
test = pd.read_csv('attack.csv')

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def format_data(data, max_features, maxlen, tokenizer=None):
    data = data.sample(frac=1).reset_index(drop=True)
    data['text'] = data['text'].apply(lambda x: str(x))

    Y = data['label'].values # 0: Real; 1: Fake
    X = data['text']

    if not tokenizer:
        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"
        tokenizer = Tokenizer(num_words=max_features, filters=filters)
        tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)

    return X, Y, tokenizer

max_features, max_len = 3000, 25
X_train, Y_train, tokenizer = format_data(train, max_features, max_len)
X_test, Y_test, _ = format_data(test, max_features, max_len, tokenizer)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500).fit(X_train, Y_train)

preds = lr.predict(X_test)

def accuracy_percentile(preds, Y_validate):
    """Return the percentage of correct predictions for each class and in total"""
    real_correct, fake_correct, total_correct = 0, 0, 0
    _, (real_count, fake_count) = np.unique(Y_validate, return_counts=True)

    for i, r in enumerate(preds):
        if r == Y_validate[i]:
            total_correct += 1
            if r == 1:
                fake_correct += 1
            else:
                real_correct += 1

    print('Real Accuracy:', real_correct/real_count * 100, '%')
    print('Fake Accuracy:', fake_correct/fake_count * 100, '%')
    print('Total Accuracy:', total_correct/(real_count + fake_count) * 100, '%')


accuracy_percentile(preds, Y_test)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

print('AUC: {}'.format(roc_auc_score(preds, Y_test)))
print('Accuracy: {}'.format(accuracy_score(preds, Y_test)))
print('Precision: {}'.format(precision_score(preds, Y_test)))
print('Recall: {}'.format(recall_score(preds, Y_test)))
print('F1: {}'.format(f1_score(preds, Y_test)))

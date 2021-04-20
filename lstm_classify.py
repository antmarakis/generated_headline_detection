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

max_features, max_len = 1000, 15
X_train, Y_train, tokenizer = format_data(train, max_features, max_len)
X_test, Y_test, _ = format_data(test, max_features, max_len, tokenizer)

from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout, LSTM
from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model

# Input shape
inp = Input(shape=(max_len,))

# Embedding and LSTM
x = Embedding(max_features, 100)(inp)
x = SpatialDropout1D(0.33)(x)
x = Bidirectional(LSTM(35, return_sequences=True))(x)

# Pooling
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])

# Output layer
output = Dense(1, activation='sigmoid')(conc)

model = Model(inputs=inp, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.load_weights('Weights/gru5.h5')
model.fit(X_train, Y_train, epochs=1, batch_size=128, verbose=1)

results = model.predict(X_test, batch_size=128, verbose=1)

def convert_to_preds(results):
    """Converts probabilistic results in [0, 1] to
    binary values, 0 and 1."""
    return [1 if r > 0.5 else 0 for r in results]

preds = convert_to_preds(results)

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
print('Precision: {}'.format(precision_score(preds, Y_test, average='macro')))
print('Recall: {}'.format(recall_score(preds, Y_test, average='macro')))
print('F1: {}'.format(f1_score(preds, Y_test, average='macro')))

import torch
import pandas as pd
import numpy as np
from simpletransformers.model import TransformerModel

torch.cuda.set_device(0)

train = pd.read_csv('defense.csv').sample(frac=1)
test = pd.read_csv('attack.csv').sample(frac=1)

train['text'] = train['text'].astype(str).str.lower()
test['text'] = test['text'].astype(str).str.lower()

model = TransformerModel('bert', 'bert-base-uncased', args={'fp16': False, 'train_batch_size': 64, 'eval_batch_size': 64, 'max_seq_length': 25})
model.train_model(train)

preds, _ = model.predict(test['text'])

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


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc

accuracy_percentile(preds, list(test['label']))
fpr, tpr, _ = roc_curve(preds, test['label'])

print('AUC: {}'.format(auc(fpr, tpr)))

print('Accuracy: {}'.format(accuracy_score(preds, test['label'])))
print('Precision: {}'.format(precision_score(preds, test['label'])))
print('Recall: {}'.format(recall_score(preds, test['label'])))
print('F1: {}'.format(f1_score(preds, test['label'])))

print('Precision: {}'.format(precision_score(preds, test['label'], average='macro')))
print('Recall: {}'.format(recall_score(preds, test['label'], average='macro')))
print('F1: {}'.format(f1_score(preds, test['label'], average='macro')))

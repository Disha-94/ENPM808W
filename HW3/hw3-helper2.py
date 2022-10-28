import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#def paren_match(page, text):
def paren_match(row):
    page = row['page']
    text = row['text']
    page = re.sub('[\(|.|\)|,|"|\t]', '', page)
    page = page.split(' ')
    page = [x.lower() for x in page]
    text = re.sub('[\(|.|\)|,|"|\t]', ' ', text)
    text = text.split(' ')
    text = [x.lower() for x in text]
    count = 0
    for item in page:
        if item in text and item not in [' ', 'at', 'a', 'of', 'the', 'on']:
            count += 1
    return count



import os
#os.chdir('/your/path/here')

train = pd.read_csv('qb.train.csv', sep=',', header=0)
test = pd.read_csv('qb.test.csv', sep=',', header=0)

print('Columns in train data: ', train.columns.values)
print(train.shape[0])
print('columns in test data: ', test.columns.values)

print(train.head())

train['paren_match'] = train.apply(paren_match, axis=1)
train['obs_len'] = train['text'].apply(len)

scaler = MinMaxScaler()
train['inlinks'] += 1
train['inlinks_log'] = np.log2(train['inlinks']) 
train[['inlinks_scaled']]  = scaler.fit_transform(train[['inlinks_log']])

print(train.head())

# one hot encoding for categorical variables

#X = pd.get_dummies(X[['category']])
train = pd.get_dummies(train, columns = ['category', 'tournaments', 'answer_type', \
'corr'])
# corr is now corr_True

#pd.set_option('display.max_colwidth',1000)
#pd.set_option('display.max_columns',1000)
#print(train.head())

features = pd.DataFrame(train, columns=['body_score', 'inlinks'])

y = train['corr_True']
#x = train.drop(['corr_True'], axis=1)
x = features

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

logreg = LogisticRegression().fit(train_x, train_y)
pred1 = logreg.predict(test_x)
cm = confusion_matrix(pred1, test_y)
print(cm)

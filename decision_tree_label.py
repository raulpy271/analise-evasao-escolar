import pandas
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

from preprocessing import columns


df = pandas.read_csv('normalized-dados-label.csv', sep=';', encoding='latin-1')
X, y = df[columns], df['LABEL_DESVINCULADO']

clf = DecisionTreeClassifier(random_state=0)

scores = cross_validate(clf, X, y, cv=8, scoring=['precision_macro', 'recall_macro'])
precision = scores['test_precision_macro']
recall = scores['test_recall_macro']
print("Precision: {:.3f} +/- {:.4f}".format(precision.mean(), precision.std()))
print("Recall: {:.3f} +/- {:.4f}".format(recall.mean(), recall.std()))

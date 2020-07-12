import pandas as pd
import numpy as np
from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
from sklearn.model_selection import train_test_split
allurlscsv = pd.read_csv('data.csv',',',error_bad_lines=False)	#reading file
df = pd.DataFrame(allurlscsv)	#converting to a dataframe

df['url'] = df['url']
df['label'] = df['label'].map({'good': 0, 'bad': 1})
url_train, url_test, label_train, label_test = train_test_split(df['url'], df['label'], test_size=0.2, random_state=42)
def perform(classifiers, vectorizers, url_train, url_test, label_train, label_test):
    max_score = 0
    max_name = 0
    for classifier in classifiers:
        for vectorizer in vectorizers:
        
            # train
            vectorize_text = vectorizer.fit_transform(url_train)
            classifier.fit(vectorize_text, label_train)

            # score
            vectorize_text = vectorizer.transform(url_test)
            score = classifier.score(vectorize_text, label_test)
            name = classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__ 
            print(name, score)
        if score > max_score:
            max_score = score
            max_name = name
    print ('===========================================')
    print ('===========================================')
    print (max_name, max_score)
    print ('===========================================')
    print ('===========================================')
classifiers = [
    MultinomialNB(),
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ]
vectorizers = [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ]
perform(
    classifiers,
    vectorizers,
   url_train, url_test, label_train, label_test
)
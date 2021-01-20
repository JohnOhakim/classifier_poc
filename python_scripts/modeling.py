import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc

train = pd.read_csv('./data/train_clean.csv')

train['is_tv'].value_counts(normalize=True)


def classifier(X, y, estimator, random_state, text, pipe_params, t_size, cross_val_size, label_fontsize, title_fontsize):
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = t_size, 
                                                        stratify=y,
                                                        random_state=random_state)
    
    
    pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('clf', estimator)
    ])
    gs = GridSearchCV(pipe, param_grid=pipe_params, cv=cross_val_size)
    gs.fit(X_train, y_train)
    print(gs.best_params_)
    print(f'The cross_val_score is: {round(gs.best_score_, 2)}')

    tn, fp, fn, tp = confusion_matrix(y_test, gs.best_estimator_.predict(X_test)).ravel()

    accuracy = round((tn + tp) / (tn + fp + fn + tp), 2)
    misclassification = round((1 - accuracy), 2)
    sensitivity = round((tp) / (tp + fn), 2)
    specificity = round((tn) / (tn + fp), 2)
    precision = round((tp) / (tp + fp), 2)

    print(f'The Accuracy is: {accuracy}')
    print(f'The Missclassification Rate is: {misclassification}')
    print(f'The Sensitivity/Recall is: {sensitivity}')
    print(f'The Specificity is: {specificity}')
    print(f'The Precision is: {precision}')

    fpr_clf, tpr_clf, _ = roc_curve(y_test, gs.best_estimator_.predict_proba(X_test)[:, 1])
    roc_auc_clf = auc(fpr_clf, tpr_clf)

    plt.figure(figsize = (10, 7))
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_clf, tpr_clf, lw=3, label='The ROC curve (area = {:0.2f})'.format(roc_auc_clf))
    plt.xlabel('False Positive Rate', fontsize=label_fontsize)
    plt.ylabel('True Positive Rate', fontsize=label_fontsize)
    plt.title(f'ROC curve {(text)}', fontsize=title_fontsize)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

    return print(classification_report(y_test, gs.best_estimator_.predict(X_test)))


classifier(train['clean_text'], train['is_tv'], 
           LogisticRegression(), 
           random_state=20720,
          text='(Logistic Regression)', pipe_params = 
           {'tvec__ngram_range': [(1, 2), (1, 3), (2,3)],
            'tvec__max_features': [200, 300, 500]},
          t_size=.40, cross_val_size=5, label_fontsize=16, title_fontsize=18  
        )

classifier(train['clean_text'], train['is_tv'], 
           GradientBoostingClassifier(warm_start=True), 
           random_state=20720,
           text='(Gradient Boosting)', pipe_params = 
           {'tvec__ngram_range': [(1, 2), (1, 3), (2, 3)],
            'tvec__max_features': [200, 300, 500],
            'clf__n_estimators': [100, 150, 200],
           'clf__learning_rate': [0.2, 0.4, 0.6],
           'clf__max_depth': [4, 6, 8]},
           t_size=.40, cross_val_size=5, label_fontsize=16, title_fontsize=18
        )

classifier(train['clean_text'], train['is_tv'],
          SVC(probability=True), random_state=20720,
          text='(Support Vector Classifier)',
          pipe_params = {
        'tvec__ngram_range': [(1, 2), (1, 3), (2, 3)],
        'tvec__max_features': [200, 300, 500],
        'clf__kernel': ['rbf', 'sigmoid', 'linear'],
        'clf__gamma': ['auto', 'scale']},
        t_size=.40, cross_val_size=5, label_fontsize=16, title_fontsize=18
    )



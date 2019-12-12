# Checking github activity to identify new open source vulnerabilities

# Task: rank a set of github events by their chance to be a real vulnerability

# Things to consider:

# 1. What preprocessing is required for the data, if any?
#     conversion of text to numeric representation and feature engineering

# 2. How to adjust for the relatively small number of positive examples?
#     a. throw some of the data
#     b. use oversampling (smote)
#     c. use balanced algorithms \ cost-sensitive learning


# 3. What features should be used?
#     url - X
#     lang - only vuln events have this information so there is no information here (and even can distract the model)
#     eventType - seems to be indicative
#     repo - X - should be further inquired - does not seem useful at the moment
#     title, description - words used in these texts will be meaningful for our objective

#
# 4. Which algorithms can work well?
#         random forest - can work well with or without oversampling
#         KNN, SVM
#         boosting algorithms

# 5. Is there an additional data that can be used to improve the model?
#     Looking at several events in a repository in order to asses vulnerability in that code
#     Information about the author of the event (account age, activity level etc
#     A pre-defined or trained list of suspicious words (for example "threat", "unsafe", "doubt", "secure", "attack" and similar)
#     Distribution of vulnerabilities among languages

# 6. How to use the programming language feature?
#    unless data is filled, there is no information in this feature (can be filled by text analysis or KNN)


#%%
import numpy as np
import pandas as pd
import scipy.stats as ss
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt
import re

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from nltk.stem.porter import PorterStemmer

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE,SMOTENC, KMeansSMOTE, SVMSMOTE
from imblearn.metrics import geometric_mean_score

from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.model_selection import cross_val_score, cross_validate, ParameterGrid

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc


#%% choose run mode
# 0 - use saved dataframes (runs immediately)
# 1 - run entire code (contains grid search and therefore will take some time)
run_flag = 1

#%% parameters for data split
test_size = 0.1
val_size = 0.2

#%%
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

#%% load data
filename = r'Tagged Data.tsv'
df = pd.read_csv(filename, sep='\t')
target_col = 'vuln'
cols = df.columns
print(cols)
print("All data: ", df.shape)

#%% use only events with title and description
df = df.dropna(subset=['title', 'description'])
print("Non-missing title and description data: ", df.shape)

#%% baseline accuracy
# temp =  df['vuln'].value_counts()
# baseline = 100 - temp[1]/(temp[0]+temp[1])*100
# print("baseline accuracy: {0:.2f}".format(baseline))

#%% explore language feature
col_name = 'lang'

plt.figure()
ax = sns.countplot(x=col_name, hue=target_col, data=df)
plt.show()

# It seems NONE of the events tagged "0" has a 'lang' value, while ALL events tagged "1" do have lang value
# Therefore, as long as this feature is not filled with valuable information, this feature can be disregarded

#%% explore eventType feature
col_name = 'eventType'
df[col_name] = df[col_name].astype('category')

plt.figure()
ax = sns.countplot(x=col_name, hue=target_col, data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.title(col_name)
plt.show()
# This seems like a valuable feature

# As there are 5 categories under the feature, consider using target encoding
encoder = ce.TargetEncoder()
encoder.fit(df[col_name], df[target_col])
df[col_name+'_enc'] = encoder.transform(df[col_name])

# Cramér’s V - based on Pearson’s Chi-Square Test
cramers_v(df[col_name], df[target_col])
cramers_v(df[col_name+'_enc'], df[target_col])

#%% explore cases where title and description are the same
col_name = 'same_title_desc'
df[col_name] = df['title']==df['description']
plt.figure()
ax = sns.countplot(x=col_name, hue=target_col, data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.title(col_name)
plt.show()
# This seems like an interesting feature

#%% preprocess text features

def prepeocess_text(col_name):
    # count number of links for each sample
    URLPATTERN = r'(https?://\S+)'
    df[col_name+'_url_count'] = df[col_name].apply(lambda x: re.findall(URLPATTERN, x)).str.len()

    col_name_clean = col_name+'_clean'
    df[col_name_clean] = df[col_name].replace(r'[^\x00-\x7F]+', ' ', regex=True)
    df[col_name_clean] = df[col_name_clean].str.lower()
    df[col_name_clean] = df[col_name_clean].replace(URLPATTERN, ' ', regex=True)
    df[col_name_clean] = df[col_name_clean].replace(r"'s", '', regex=True)
    df[col_name_clean] = df[col_name_clean].replace(r"'", '', regex=True)
    df[col_name_clean] = df[col_name_clean].replace('<.*?>', ' ', regex=True)
    df[col_name_clean] = df[col_name_clean].replace(r'[^\w\s]', ' ', regex=True)
    df[col_name_clean] = df[col_name_clean].replace(r'\d+', ' ', regex=True)
    df[col_name_clean] = df[col_name_clean].replace(r'_', ' ', regex=True)

    # tokenization
    df[col_name_clean] = df[col_name_clean].apply(word_tokenize)

    # remove stop words
    stop = stopwords.words('english')
    df[col_name_clean] = df[col_name_clean].apply(lambda x: [item for item in x if item not in stop])
    df[col_name_clean] = df[col_name_clean].apply(lambda x: [item for item in x if len(item)>1])

    # stemming (another approach: lemmatization)
    stem = PorterStemmer()  # the least aggressive stemmer
    df[col_name_clean] = df[col_name_clean].apply(lambda x: [stem.stem(y) for y in x])

    # count number of words for each sample
    df[col_name+'_word_count'] = df[col_name_clean].apply(lambda x: len(x))

    # join back to text
    df[col_name_clean] = df[col_name_clean].apply(lambda x: " ".join(x))

if run_flag:
    # preprocess title text
    col_name = 'title'
    prepeocess_text(col_name)

    # preprocess description text
    col_name = 'description'
    prepeocess_text(col_name)

    df.to_pickle('df_preprocessed_text.pkl')
else:
    df = pd.read_pickle("df_preprocessed_text.pkl")


#%% train-val-test split
X_train, X_test, y_train, y_test = train_test_split(df[['eventType', 'eventType_enc',
                                                        'title', 'title_clean', 'title_url_count', 'title_word_count',
                                                        'description', 'description_clean', 'description_url_count', 'description_word_count',
                                                        'same_title_desc']],
                                                    df['vuln'],
                                                    test_size=test_size,
                                                    random_state=30)

X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=val_size,
                                                  random_state=30)


#%% convert text to features using tf-idf
# Parameters:
# ngram_range: We want to consider both unigrams and bigrams.
# max_df: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold
# min_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
# max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

def fit_tfidf(docs, text_source_name):
    tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=ngram_range, stop_words=None, lowercase=False, max_df=max_df,
                            min_df=min_df, max_features=max_features, norm='l2', sublinear_tf=True)
    features = tfidf.fit_transform(docs).toarray()
    df_features = pd.DataFrame(features)
    df_features.columns = [s + '_' + text_source_name for s in tfidf.get_feature_names()]
    return df_features


def get_best_words(features, y, N):
    features_chi2 = chi2(features, y)
    indices = np.argsort(features_chi2[0])
    feature_names = features.columns[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("")
    print("Most correlated unigrams:\n{}".format(', '.join(list(reversed(unigrams))[:N])))
    print("")
    print("Most correlated bigrams:\n{}".format(', '.join(list(reversed(bigrams))[:N])))

if run_flag:

    # convert title text to features
    ngram_range = (1, 2)
    min_df = 1
    max_df = 1.0
    max_features = 300

    # fit over train data and transform
    features_title_train = fit_tfidf(X_train['title_clean'], 'title')
    get_best_words(features_title_train, y_train, 10)

    # transform data
    features_title_val = fit_tfidf(X_val['title_clean'], 'title')
    features_title_test = fit_tfidf(X_test['title_clean'], 'title')

    # convert description text to features
    ngram_range = (1, 2)
    min_df = 1  # do not ignore rare words as they might have meaningful information for anomalies
    max_df = 1.0
    max_features = 300

    # fit over train data and transform
    features_desc_train = fit_tfidf(X_train['description_clean'], 'desc')
    get_best_words(features_desc_train, y_train, 10)
    # some expected words were found: memori leak, consol log, infinit loop, node modul, script, secur, vulner, error

    # transform data
    features_desc_val = fit_tfidf(X_val['description_clean'], 'desc')
    features_desc_test = fit_tfidf(X_test['description_clean'], 'desc')

    # save dataframes
    features_title_train.to_pickle(r'features_title_train.pkl')
    features_title_val.to_pickle(r'features_title_val.pkl')
    features_title_test.to_pickle(r'features_title_test.pkl')
    features_desc_train.to_pickle(r'features_desc_train.pkl')
    features_desc_val.to_pickle(r'features_desc_val.pkl')
    features_desc_test.to_pickle(r'features_desc_test.pkl')

else:
    # load dataframes
    features_title_train = pd.read_pickle(r'features_title_train.pkl')
    features_title_val = pd.read_pickle(r'features_title_val.pkl')
    features_title_test = pd.read_pickle(r'features_title_test.pkl')
    features_desc_train = pd.read_pickle(r'features_desc_train.pkl')
    features_desc_val = pd.read_pickle(r'features_desc_val.pkl')
    features_desc_test = pd.read_pickle(r'features_desc_test.pkl')


#%%
print("")

full_df_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(features_title_train), pd.DataFrame(features_desc_train)], axis=1)
print("train data:")
print(full_df_train.shape)

full_df_val = pd.concat([X_val.reset_index(drop=True), pd.DataFrame(features_title_val), pd.DataFrame(features_desc_val)], axis=1)
print("val data:")
print(full_df_val.shape)

full_df_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(features_title_test), pd.DataFrame(features_desc_test)], axis=1)
print("test data:")
print(full_df_test.shape)

#%% organize data
col_to_remove = ['eventType', 'title', 'description', 'title_clean', 'description_clean']

full_df_train_filt = full_df_train.drop(columns=col_to_remove)
full_df_val_filt = full_df_val.drop(columns=col_to_remove)
full_df_test_filt = full_df_test.drop(columns=col_to_remove)


#%% feature selection
# number of features vs auc was examined to determine number of features
k_list = [10, 50, 100, 200, 250, 300, 400, 500, 600]
# auc over train data: 0.75, 0.77, 0.79, 0.81, 0.82, 0.82, 0.83, 0.83, 0.83

k = 250
fs = SelectKBest(chi2, k=k)
fs.fit(full_df_train_filt, y_train)
cols = fs.get_support(indices=True)

df_train = full_df_train_filt.iloc[:, cols]
df_val = full_df_val_filt.iloc[:, cols]
df_test = full_df_test_filt.iloc[:, cols]

#%% oversmapling
sm = SMOTE(random_state=8)
# sm = SVMSMOTE(random_state=8)

df_train_resampled, y_train_resampled = sm.fit_resample(df_train, y_train)

#%%
def print_full_results(y, y_pred, y_pred_prob, title):
    # print("")
    # print("The accuracy is: ")
    # print("{0:.2f}".format(accuracy_score(y, y_pred) * 100))
    # print(" ")
    # print("The balanced accuracy is: ")
    # print("{0:.2f}".format(balanced_accuracy_score(y, y_pred) * 100))

    cm = confusion_matrix(y, y_pred)
    print(" ")
    print("The confusion matrix is: ")
    print(cm)
    print(" ")
    print(cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis])
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix - ' + title)
    plt.show()

    fpr, tpr, thresholds = roc_curve(y, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=1)
    plt.title('ROC (AUC = %0.2f) ' % (roc_auc) + title)
    plt.xlabel('FPR')
    plt.ylabel('TPR (recall)')
    plt.show()
    print(" ")
    print("The AUC is: ")
    print("{0:.2f}".format(roc_auc*100))

    print(" ")
    print("Classification report")
    print(classification_report(y, y_pred))

    return fpr, tpr


#%% grid search

if 0:
    # parameters for grid search
    bootstrap = [True, False]
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 30, num = 30)]
    max_features = ['sqrt', None]
    min_samples_leaf = [int(x) for x in np.linspace(start = 100, stop = 700, num = 7)]
    min_samples_split = [int(x) for x in np.linspace(start = 100, stop = 700, num = 7)]
    n_estimators = [int(x) for x in np.linspace(start = 2, stop = 30, num = 15)]
    criterion = ['gini', 'entropy']
    class_weight = None #'balanced_subsample' , 'balanced'
    param_grid = {
        'bootstrap': bootstrap,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators,
        'criterion' : criterion
    }

    # data
    x = df_train_resampled
    y = y_train_resampled

    # model
    mdl = RandomForestClassifier(random_state=8)
    # mdl = svm.SVC(probability=True)
    # mdl = BalancedBaggingClassifier(random_state=8)

    # grid search
    best_score = 0
    for g in ParameterGrid(param_grid):
        mdl.set_params(**g)
        mdl.fit(x, y)
        score = f1_score(y_val, mdl.predict(df_val))
        if score > best_score:
            best_score = score
            best_grid = g

    # grid search results
    print("")
    print("The best hyperparameters from Grid Search are:")
    print(best_grid)
    print("")
    print("The validation f1 value of a model with these hyperparameters is:")
    print(best_score)

    # train best model
    mdl.set_params(**best_grid)
    mdl.fit(x, y)


#%% train a chosen model

# params
bootstrap = True
max_depth = 3
max_features = None
min_samples_leaf = 500
min_samples_split = 500
n_estimators = 6
criterion = 'entropy'

# data
x = df_train_resampled
y = y_train_resampled

# model
mdl = RandomForestClassifier(bootstrap=bootstrap, max_depth=max_depth, max_features=max_features,
                             min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                             n_estimators=n_estimators, criterion=criterion, random_state=8)
mdl.fit(x, y)

# train results
y_pred = mdl.predict(df_train)
y_pred_prob = mdl.predict_proba(df_train)
score_train = f1_score(y_train, y_pred)
print("")
print("")
print("Training results: ")
print("")
print("The f1 score is: ")
print(score_train)
fpr_train, tpr_train = print_full_results(y_train, y_pred, y_pred_prob, 'training set')

# val results
y_pred = mdl.predict(df_val)
y_pred_prob = mdl.predict_proba(df_val)
score_val = f1_score(y_val, y_pred)
print("")
print("")
print("Validation results: ")
print("")
print("The f1 score is: ")
print(score_val)
fpr_val, tpr_val = print_full_results(y_val, y_pred, y_pred_prob, 'validation set')

# test results
y_pred = mdl.predict(df_test)
y_pred_prob = mdl.predict_proba(df_test)
score_test = f1_score(y_test, y_pred)
print("")
print("")
print("Test results: ")
print("")
print("The f1 score is: ")
print(score_test)
fpr_test, tpr_test = print_full_results(y_test, y_pred, y_pred_prob, 'test set')

# plot roc curves
plt.figure()
plt.plot(fpr_train, tpr_train, lw=1, alpha=1)
plt.plot(fpr_val, tpr_val, lw=1, alpha=1)
plt.plot(fpr_test, tpr_test, lw=1, alpha=1)
plt.legend(['train', 'val', 'test'])
plt.title('Train, Val and Test ROC')
plt.xlabel('FPR')
plt.ylabel('TPR (recall)')
plt.show()


#%% create results dataframe
df_test_results = full_df_test.loc[:, ['title', 'description', 'eventType_enc']]
df_test_results.loc[:, 'vuln'] = np.array(y_test)
df_test_results.loc[:, 'prob'] = y_pred_prob[:, 1]
df_test_results = df_test_results.sort_values(by=['prob'], ascending=False)
print("")
print("Probabilities:")
print(df_test_results)

# prob counts
probs_count = df_test_results.prob.value_counts().to_frame(name='prob_count').reset_index().rename(columns={"index": "prob"})
print("The results are a somewhat 'partial' ranking, as same probabilities are repeating for several samples:")
print(probs_count.sort_values(by='prob', ascending=False)[:10])

#%% important features
from tabulate import tabulate
headers = ["name", "score"]
values = sorted(zip(df_train.columns, mdl.feature_importances_), key=lambda x: x[1] * -1)
print("")
print("Important features:")
print(tabulate(values[:10], headers[:10], tablefmt="plain"))


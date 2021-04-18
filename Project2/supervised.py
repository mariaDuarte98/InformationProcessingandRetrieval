import threading

from main import read_xml_files, red_qrels_file, read_topics_file
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

numTopics = 5
vectorizer = TfidfVectorizer()
pca_model = PCA(n_components=20)
lock = threading.Lock()


def training(q, Dtrain, Rtrain, args='NaiveBayes'):
    collection = []
    doc_relevance = []
    for doc in Dtrain.keys():
        doc_sentences = ""
        for token in Dtrain[doc]:
            doc_sentences += " " + token
        collection.append(doc_sentences)
        if doc in Rtrain[q]:
            doc_relevance.append(1)
        else:
            doc_relevance.append(0)

    X = vectorizer.fit_transform(collection).toarray()
    #  X = pca_model.fit_transform(X)
    #X = pca.transform(X)
    #print(X)
    if args == 'NaiveBayes':
        gnb = GaussianNB()
        return gnb.fit(X, doc_relevance)

    elif args == 'RandomForest':
        clf = RandomForestClassifier() # entropy also ; numer of tree default is 100, change parameters for the report
        # # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
        random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        return random.fit(X, doc_relevance)
        #return clf.fit(X, doc_relevance)


def classify(d, M, args=None):
    s = ""
    for word in d:
        s += word + " "
    x_test = vectorizer.transform([s]).toarray()
   # pca = pca_model.fit(x_test)
    #   x_test = pca_model.transform(x_test)
    if args == 'prob':
        return M.predict_proba(x_test) # probabilistic output
    else:
        return M.predict(x_test)

def rank_docs(D, M, p):
    prob_docs = {}
    for doc in D.keys():
        prob = classify(D[doc], M, 'prob')[0][0]
        prob_docs[doc] = prob
    print(prob_docs)

    return sorted(prob_docs, key=prob_docs.get, reverse=True)[:p]

def evaluate(q, Dtest, Rtest, args=None):
    classification_model = training(q, train_xmls, q_rels_train_dict, 'NaiveBayes')
    # evaluate with relevance feedback
    labels_pred, labels_test = [], []
    for doc in Dtest.keys():
        label = classify(Dtest[doc], q, classification_model)[0]
        labels_pred.append(label)
        if doc in Rtest[q]:
            labels_test.append(1)
        else:
            labels_test.append(0)
            
    if 1 not in labels_pred:
      return 0, 1 , 0, 0, 0
    accuracy = accuracy_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred)
    recall = recall_score(labels_test, labels_pred)
    f1 = f1_score(labels_test, labels_pred)

    return accuracy, 1 - accuracy, precision, recall, f1  # accuracy and error rate for all topics

def worker(q):

    result = evaluate(q, test_xmls, q_rels_test_dict, args=None)

    lock.acquire()
    print("Topic " + q + " =>")
    print(result)
    print()
    lock.release()
    accuracies.append(result)


D_PATH = 'rcv1/'
q_topics_dict = read_topics_file()  # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_train_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic
q_rels_test_dict = red_qrels_file("qrels.test.txt")  # dictionary with topic id: relevant document id ,for each topic
train_xmls, test_xmls, _ = read_xml_files(D_PATH)

'''
thread_list = []
for q in list(q_rels_test_dict.keys())[:numTopics]:
    x = threading.Thread(target=worker, args=(q,))
    thread_list.append(x)
    x.start()

for x in thread_list:
    x.join()
'''

'''
for test_file in test_xmls.keys():
    result = classify(test_xmls[test_file], q, classification_model)
'''

"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@   Instituto Superior Tecnico  @
@@      PRI - 1st Delivery     @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@    Dinis Araújo - 86406     @@
@@    Inês Lacerda - 86436     @@
@@    Maria Duarte - 86474     @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
import datetime
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from sklearn.feature_extraction import text
import nltk
import pandas
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import ml_metrics
from rank_bm25 import BM25Okapi

from PyDictionary import PyDictionary


# Just some input variable to run our experiments with the analyses.py file
weights = [int(input("First weight: ")), int(input("second weight: ")), int(input("third weight: "))]
stop_words_flag = input("True or False? ")
ranking_k = int(input("Integer between 1 and 1000: "))
delete_irrel = input("Want to to delete irrelevant:True or False? ")
syn_flag = input("Want to add synonyms:True or False? ")

results_folder = str(weights[0]) + str(weights[1]) + str(weights[2]) + "SW" + str(stop_words_flag) + \
                 "docs" + str(ranking_k) + str(delete_irrel) + str(syn_flag)

# Creating folders where we save the results
if not os.path.isdir("./Results/"):
    os.mkdir("./Results/")
if not os.path.isdir("./Results/" + results_folder):
    os.mkdir("./Results/" + results_folder)


# Creating topic class to represent its structure
class Topic:
    def __init__(self, title, desc, narr):
        self.title = title
        self.desc = desc
        self.narr = narr


# function that processes qrels file
def red_qrels_file():
    rels_dict = {}
    with open(Q_RELS_TEST) as file:
        for line in file:
            topic_num, doc_id, relevante_bool = line.split(' ')
            relevante_bool = relevante_bool.replace('\n', '')
            if topic_num in rels_dict:
                if relevante_bool == '1':
                    rels_dict[topic_num].append(doc_id)
            else:
                if relevante_bool == '1':
                    rels_dict[topic_num] = [doc_id]
    return rels_dict


# function that processes topics file
def read_topics_file():
    topics_dic = {}
    with open(Q_TOPICS_PATH, 'r') as file:
        for line in file:
            if '<num>' in line:
                num = (line.replace('<num> Number: ', '')).replace('\n', '')
                num = num.replace(" ", "")
            elif '<title>' in line:
                title = line.replace('<title>', '')
            elif '<desc>' in line:
                desc = ""
                line = file.readline()
                while '<narr>' not in line:
                    if len(line) > 1:
                        desc += " " + line
                    line = file.readline()
                narr = ""
                line = file.readline()
                while '</top>' not in line:
                    if len(line) > 1:
                        narr += " " + line
                    line = file.readline()
                if delete_irrel == 'True':
                    narr = update_narrative(narr)
                topics_dic[num] = Topic(title, desc, narr)
    return topics_dic


#  Function responsible of preprocessing all the tokens:
#  punctuation, lower casing, stopwords, stemming
def preprocessing(content, content_type="document"):
    # punctuation
    content = re.sub(r'\W', ' ', content)
    # lower casing
    tokens = nltk.word_tokenize(content.lower())
    # stop words
    if stop_words_flag == 'True':
        stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
    else:
        stop_words = []
    # stemming
    ps = PorterStemmer()
    preprocessed_tokens = []
    dictionary=PyDictionary()

    for t in tokens:
        if t not in stop_words: # and len(t) > 3:
            preprocessed_tokens.append(ps.stem(t))
            if content_type == "topic" and syn_flag == "True":
                synonyms = dictionary.synonym(t)
                if synonyms:
                    for syn in synonyms:
                        syn_tokens = nltk.word_tokenize(syn.lower())
                        [preprocessed_tokens.append(ps.stem(token)) for token in syn_tokens if token not in stop_words]

    return preprocessed_tokens


#  Function responsible for deleting irrelevant tokens from topic narrative
def update_narrative(narrative):
    narrative_list = narrative.split(".")
    final_narrative = ""
    for sent in narrative_list:
        if "irrelevant" not in sent:
            final_narrative += sent
    return final_narrative


# Function responsible for reading documents, and saving the collection
def read_xml_files():
    folders = os.listdir(D_PATH)
    train_xmls = {}
    test_xmls = {}
    for folder in folders:
        if not os.path.isfile(os.path.join(D_PATH, folder)):
            xml_file_names = os.listdir(D_PATH + folder + "/")
            for xml_file_name in xml_file_names:
                if os.path.isfile(os.path.join(D_PATH, folder + "/" + xml_file_name)) and xml_file_name.find(
                        ".xml") != -1:
                    xml_file = ET.parse(D_PATH + folder + "/" + xml_file_name)
                    year, month, day = [int(x) for x in
                                        xml_file.getroot().attrib.get('date').split(
                                            '-')]
                    date = datetime.date(year, month, day)

                    document = ''
                    for tag in ['headline', 'byline', 'dateline']:
                        for content in xml_file.getroot().iter(tag):
                            if content.text:
                                document += ' ' + content.text
                    for content in xml_file.getroot().iter('text'):
                        for paragraph in content:
                            document += ' ' + paragraph.text
                    if date <= DATE_TRAIN_UNTIL:
                        train_xmls[xml_file.getroot().attrib.get('itemid')] = preprocessing(document)
                    else:
                        test_xmls[xml_file.getroot().attrib.get('itemid')] = preprocessing(document)

    return train_xmls, test_xmls


# Function that creates the tfidf and tfidf matrix
def tfidf_creation(topic_text, I, compare=False):
    processed = list(dict.fromkeys(topic_text))
    collection = []

    if compare:
        collection.append(' '.join(topic_text))

    for doc in I.keys():
        doc_sentences = ""
        for token in I[doc].keys():
            for i in range(I[doc][token]):
                doc_sentences += " " + token
        collection.append(doc_sentences)

    tfidf = TfidfVectorizer(vocabulary=processed)
    tfidf_matrix = tfidf.fit_transform(collection)

    return tfidf, tfidf_matrix


#########################
#        Indexing       #
#########################
def indexing(D, args=None):
    # @input D and optional set of arguments on text preprocessing @behavior preprocesses each document in D and
    # builds an efficient inverted index (with the necessary statistics for the subsequent functions)
    # @output tuple with the inverted index I, indexing time and space required
    doc_token_count = {}
    start = time.time()
    for document in D.keys():
        for token in D[document]:
            if document in doc_token_count:
                if token in doc_token_count[document]:
                    doc_token_count[document][token] += 1
                else:
                    doc_token_count[document][token] = 1
            else:
                doc_token_count[document] = {}
                doc_token_count[document][token] = 1
    end = time.time()
    return doc_token_count, end - start, sys.getsizeof(doc_token_count)


#########################
#  Extract Topic Query  #
#########################
def extract_topic_query(q, I, k=3, args=None):
    # @input topic q ∈ Q (identifier), inverted index I, number of top terms for the topic (k), and optional
    # arguments on scoring
    # @behavior selects the top-k informative terms in q against I using parameterizable scoring
    # @output list of k terms (a term can be either a word or phrase)

    tfidf, tfidf_matrix = tfidf_creation(q, I)  # creating the tfif
    features_names = tfidf.get_feature_names()

    dense = tfidf_matrix.todense()
    dense_list = dense.tolist()
    df = pandas.DataFrame(dense_list, columns=features_names)
    s_mean = df.mean()  # by doing the mean of every columns in a dataframe we get a series
    s_maximum = s_mean.nlargest(k)  # getting the highest k elements from series

    return list(s_maximum.keys())  # returning the tokens that have the highest relevance


#########################
#     Boolean Query     #
#########################
def boolean_query(q, I, k=3, args=None):
    # @input topic q (identifier), number of top terms k, and index I
    # @behavior maps the inputted topic into a simplified Boolean query using
    # extract topic query and then search for matching* documents using the Boolean IR model
    # @output the filtered collection, specifically an ordered list of document identifiers
    k_terms = extract_topic_query(q, I, k)
    docs_id = {}
    all_docs = []
    all_docs_scores = {}
    for doc in I.keys():
        common_terms = len(list(set(I[doc].keys()).intersection(k_terms)))
        all_docs.append(doc)
        if common_terms >= math.ceil(0.8 * k):
            docs_id[doc] = common_terms
        all_docs_scores[doc] = common_terms
    return sorted(docs_id, key=docs_id.get, reverse=True), all_docs, all_docs_scores


#########################
#        Ranking        #
#########################
def ranking(q, I, p=ranking_k, args=None):
    # @input topic q ∈ Q (identifier), number of top documents to return (p), index I,
    # optional arguments on IR models
    # @behavior uses directly the topic text (without extracting simpler queries) to rank
    # documents in the indexed collection using the vector space model or probabilistic retrieval model
    # @output ordered set of top-p documents, specifically a list of pairs – (document
    # identifier, scoring) – ordered in descending order of score

    _, tfidf_matrix = tfidf_creation(q, I, True)

    cosine_similarities = list(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten())

    dic = {}
    count = 0
    for cosine in cosine_similarities[1:]:
        dic[list(I.keys())[count]] = cosine
        count += 1

    docs_id_sorted = sorted(dic, key=dic.get, reverse=True)

    docs = []
    [docs.append((cosine, dic[cosine])) for cosine in docs_id_sorted]

    return docs[:p], docs_id_sorted, dic  # returning (docId,docScore) , docId


#########################
#         BM25          #
#########################

# NOTE: We use BM25 as an optional ranking model
def bm25(topic_text, I, p=ranking_k, args=None):
    col = []
    for key in test_xmls.keys():
        col.append(test_xmls[key])

    bm25 = BM25Okapi(col)
    doc_scores = bm25.get_scores(topic_text)

    dic = {}
    count = 0
    for score in doc_scores:
        dic[list(I.keys())[count]] = score
        count += 1
    bm25_sorted = sorted(dic, key=dic.get, reverse=True)

    docs = []
    [docs.append((score, dic[score])) for score in bm25_sorted]

    return docs[:p], bm25_sorted, dic  # returning (docId,docScore) , docId of all docs for rrf model

##############################
#  Functions for evaluation  #
##############################


# Function that evaluates boolean query
def eval_boolean_query(t, rel_docs, relevant_len, docs_id):
    f_score = 0

    relevant_retrieved = [doc for doc in docs_id if doc in rel_docs[t]]

    if docs_id:  # there are some topics for each the boolean query might not return anything
        precision = len(relevant_retrieved) / len(docs_id)
    else:
        precision = 0
    recall = len(relevant_retrieved) / relevant_len
    # the weighted harmonic mean of precision and recall
    if precision + recall != 0:
        f_score = (0.5**2 + 1) * ((precision * recall) / ((0.5**2 * precision) + recall))

    return precision, recall, f_score


# Function to calculate precision and recall of ranked documents
def precision_recall_rank(docs, rel_test_topic, relevant_len, model):
    count = 0
    precision_rank = []
    recall_rank = []
    docs_ids = []

    for i in range(1, len(docs) + 1):
        if model == 'rrf' or model == 'comb_sum' or model == 'comb_mnz':
            doc = docs[i - 1]
        else:
            doc = docs[i - 1][0]
        docs_ids.append(doc)
        if doc in rel_test_topic:
            count += 1
        precision_rank.append(count / i)
        recall_rank.append(count / relevant_len)

    precision_rank2 = precision_rank.copy()

    # interpolation...
    i = len(recall_rank) - 2
    while i >= 0:
        if precision_rank[i + 1] > precision_rank[i]:
            precision_rank[i] = precision_rank[i + 1]
        i = i - 1

    fig, ax = plt.subplots()
    for i in range(len(recall_rank) - 1):
        ax.plot((recall_rank[i], recall_rank[i]), (precision_rank[i], precision_rank[i + 1]), 'k-', label='',
                color='red')  # vertical
        ax.plot((recall_rank[i], recall_rank[i + 1]), (precision_rank[i + 1], precision_rank[i + 1]), 'k-', label='',
                color='red')  # horizontal

    ax.plot(recall_rank, precision_rank2, 'k--', color='blue')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.savefig("./Results/" + results_folder + "/plot" + topic + "_" + model + ".png")
    fig.clf()

    return precision_rank, recall_rank, docs_ids


# Auxiliary Function that calculates BPREF Measure
def counter(ranked_docs, doc_id, rel_test):
    count = 0
    for doc in ranked_docs:
        if doc_id == doc:
            return count
        elif doc not in rel_test:
            count += 1
    return count


# Function that calculates BPREF Measure
def bpref(docs_ids, rel_test_topic, relevant_len, relevant_retrieved_rank):
    bPref = 0
    for doc in docs_ids:
        if doc in rel_test_topic:
            count = counter(docs_ids, doc, rel_test_topic)
            if min(relevant_len, len(docs_ids) - len(relevant_retrieved_rank)) == 0:
                bPref += 1
            else:
                bPref += 1 - (count / min(relevant_len, len(docs_ids) - len(relevant_retrieved_rank)))
    bPref /= relevant_len

    return bPref


# Function that calculates the rrf score
def rrf_score(docs_rank, docs_bm, docs_bool, p=ranking_k):
    rrf_doc_scores = {}
    for doc in I.keys():
        rrf_doc_scores[doc] = (1 / (50 + docs_rank.index(doc))) + (1 / (50 + docs_bm.index(doc)))

    return sorted(rrf_doc_scores, key=rrf_doc_scores.get, reverse=True)[:p]


# Function that calculates the combSum score
def comb_sum_score(docs_rank_score, docs_bm_score, docs_bool_score, p=ranking_k):
    comb_sum_scores = {}
    for doc in I.keys():
        comb_sum_scores[doc] = docs_rank_score[doc] + docs_bm_score[doc] + docs_bool_score[doc]

    return sorted(comb_sum_scores, key=comb_sum_scores.get, reverse=True)[:p]


# Function that calculates the combMNZ score
def comb_mnz_score(docs_rank_score, docs_rank, docs_bm_score, docs_bm, docs_bool_score, docs_bool, p=ranking_k):
    comb_mnz_scores = {}
    for doc in I.keys():
        comb_mnz_scores[doc] = ((1 + docs_rank.index(doc))*docs_rank_score[doc] + (docs_bm.index(doc) + 1)*docs_bm_score[doc] +
                                (docs_bool.index(doc) + 1)*docs_bool_score[doc])

    return sorted(comb_mnz_scores, key=comb_mnz_scores.get, reverse=True)[:p]


# Function that evaluates ranking
def eval_ranking(rel_test_topic, relevant_len, docs, model):
    # Precision, Recall
    precision, recall, docs_ids = precision_recall_rank(docs, rel_test_topic, relevant_len, model)
    relevant_retrieved_rank = [doc for doc in docs_ids if doc in rel_test_topic]

    # MAP
    map_results_rank = []
    print("\nMAP:")
    for k in range(1, len(docs_ids)):
        map_results_rank.append(ml_metrics.mapk(rel_test_topic, docs_ids, k))
        print(ml_metrics.mapk(rel_test_topic, docs_ids, k))

    # BPREF
    bPref = bpref(docs_ids, rel_test_topic, relevant_len, relevant_retrieved_rank)

    return precision, recall, map_results_rank, bPref


def evaluation(t, topic_text, Qtest, Rtest, Dtest, args=None):
    # @input set of topics Qtest ⊆ Q, document collection Dtest, relevance feedback
    # Rtest, arguments on text processing and retrieval models
    # @behavior uses the aforementioned functions of the target IR system to test simple
    # retrieval (Boolean querying) tasks or ranking tasks for each topic q ∈
    # Qtest, and comprehensively evaluates the IR system against the available
    # relevance feedback
    # @output extensive evaluation statistics for the inputted queries, including recall-and-precision curves at
    # different output sizes, MAP, BPREF analysis, cumulative gains and efficiency

    relevant_len = len(Rtest[topic])
    # Boolean query
    docs_bool, all_docs_id_bool, all_docs_scores_bool = boolean_query(topic_text, I)

    precision, recall, f_score = eval_boolean_query(t, Rtest, relevant_len, docs_bool)
    boolean = (precision, recall, f_score)

    # Ranking
    docs_rank, all_docs_ids_rank, docs_scores_rank = ranking(topic_text, I)
    svm = eval_ranking(Rtest[t], relevant_len, docs_rank, "rank")

    # BM25
    docs_bm, all_docs_ids_bm, docs_scores_bm = bm25(topic_text, I)
    bm_model = eval_ranking(Rtest[t], relevant_len, docs_bm, "bm")

    # RRF
    docs_rrf = rrf_score(all_docs_ids_rank, all_docs_ids_bm, all_docs_id_bool)
    rrf_model = eval_ranking(Rtest[t], relevant_len, docs_rrf, "rrf")

    # CombSum
    docs = comb_sum_score(docs_scores_rank, docs_scores_bm, all_docs_scores_bool)
    comb_sum_model = eval_ranking(Rtest[t], relevant_len, docs, "comb_sum")

    # CombMNZ
    docs = comb_mnz_score(docs_scores_rank,all_docs_ids_rank, docs_scores_bm, all_docs_ids_bm, all_docs_scores_bool, all_docs_id_bool)
    comb_mnz_model = eval_ranking(Rtest[t], relevant_len, docs, "comb_mnz")

    return boolean, svm, bm_model, rrf_model, comb_sum_model, comb_mnz_model


topics_ids = list(range(1, 101)) # There are 100 topics

Q_TOPICS_PATH = "topics.txt"
Q_RELS_TEST = "qrels.test.txt"

q_topics_dict = read_topics_file()   # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_test_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic

for topic_num in topics_ids:  # q_topics_dict.keys():
    if topic_num <= 9:
        topic = "R10" + str(topic_num)  # for example for 3, we have topic R103
    elif topic_num != 100:
        topic = "R1" + str(topic_num)   # for example for 14 we have topic R114
    else:
        topic = "R200"              # for 100 we have topic R200
    D_PATH = "rcv1_r" + str(topic_num) + "/"

    DATE_TRAIN_UNTIL = datetime.date(1996, 9, 30)
    train_xmls, test_xmls = read_xml_files()

    I, time_complexity, memory_usage = indexing(test_xmls)

    topic_content = q_topics_dict[topic]
    topic_string = ''
    for i in range(0,weights[0]):
        topic_string += topic_content.title + ' '
    for i in range(0, weights[1]):
        topic_string += topic_content.desc + ' '
    for i in range(0, weights[2]):
        topic_string += topic_content.narr + ' '
    topic_processed = preprocessing(topic_string, "topic")

    boolean_model, space_vector_model, bm, rrf, comb_sum, comb_mnz = evaluation(topic, topic_processed, q_rels_test_dict.keys(), q_rels_test_dict, test_xmls)

    print("#########results######")

    print("\nboolean model:(precision, recall, f_score)")
    print(boolean_model)

    print("\nspace_vector_model:(precision, recall, map, bPref)")
    print(space_vector_model)

    print("\nbm25 model:(precision, recall, map, bPref)")
    print(bm)

    print("\nrrf model:(precision, recall, map, bPref)")
    print(rrf)

    print("\ncombSum model:(precision, recall, map, bPref)")
    print(comb_sum)

    print("\ncombMNZ model:(precision, recall, map, bPref)")
    print(comb_mnz)

    # writing into corresponding topic file all results
    f = open("./Results/" + results_folder + "/" + topic + ".txt", "w")
    f.write("\n" + str(boolean_model))
    f.write("\n" + str(space_vector_model))
    f.write("\n" + str(bm))
    f.write("\n" + str(rrf))
    f.write("\n" + str(comb_sum))
    f.write("\n" + str(comb_mnz))
    f.close()

print('Done!')


# todo text processing options
# todo IR models
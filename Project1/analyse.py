import matplotlib.pyplot as plt
from main import *
import pandas


#topics_ids = list(range(1, 101))
boolean_model_results = [[], [], []]  # position 0: precision position 1: recall position 2: f measure
space_vector_results = [[], [], []]  # position 0: avg precision position 1: avg recall position 2: bpref
bm25_results = [[], [], []]  # position 0: avg precision position 1: avg recall position 2: bpref
rrf_results = [[], [], []]
comb_results = [[], [], []]
comb_mnz_results = [[], [], []]
topics = []

folder_name = "./Results/" + results_folder + "/"
print("kjhgfdfghjkjhg")
print(results_folder)

avg_precisions_topics = []
for topic_num in topics_ids:  # q_topics_dict.keys():
    if topic_num != 100 and topic_num <= 9:
        topic = "R10" + str(topic_num)  # for example for 3, we have topic R103
    elif topic_num != 100:
        topic = "R1" + str(topic_num)   # for example for 14 we have topic R114
    else:
        topic = "R200"              # for 100 we have topic R200

    topics.append(topic)
    f = open(folder_name + topic + ".txt", "r")
    lines = f.readlines()
    f.close()

    # Getting All Boolean Model Results.
    # Written in file with the following format: (Precision, Recall, F-Measure)
    boolean_model = lines[1].replace("\n", "").replace("(", "").replace(")", "").replace(" ", "").split(",")

    boolean_precision = float(boolean_model[0])
    boolean_recall = float(boolean_model[1])
    boolean_f = float(boolean_model[2])

    boolean_model_results[0].append(boolean_precision)
    boolean_model_results[1].append(boolean_recall)
    boolean_model_results[2].append(boolean_f)


    # Getting All Space vector Model Results.
    # Written in file with the following format: (Precisions, Recalls, Docs_Ids, B_PREF)
    space_vector_model = lines[2].replace("(", "").replace(")", "").replace("[", "").split("],")

    svm_precisions = space_vector_model[0].split(",")
    last_precision_svm = svm_precisions[-1]
    svm_recalls = space_vector_model[1].split(",")
    last_recall_svm = svm_recalls[-1]
    b_pref_svm = float(space_vector_model[3])

    space_vector_results[0].append(float(last_precision_svm))
    space_vector_results[1].append(float(last_recall_svm))
    space_vector_results[2].append(b_pref_svm)

    # Getting All BM25 Results.
    # Written in file with the following format: (Precisions, Recalls, Docs_Ids, B_PREF)
    bm25_model = lines[3].replace("(", "").replace(")", "").replace("[", "").split("],")
    bm25_precisions = bm25_model[0].split(",")
    last_precision_bm = bm25_precisions[-1]
    bm25_recalls = bm25_model[1].split(",")
    last_recall_bm = bm25_recalls[-1]
    b_pref_bm = float(bm25_model[3])

    bm25_results[0].append(float(last_precision_bm))
    bm25_results[1].append(float(last_recall_bm))
    bm25_results[2].append(b_pref_bm)

    # Getting All RRF Results.
    # Written in file with the following format: (Precisions, Recalls, Docs_Ids, B_PREF)
    rrf_model = lines[4].replace("(", "").replace(")", "").replace("[", "").split("],")

    rrf_precisions = rrf_model[0].split(",")
    last_precision_rrf = rrf_precisions[-1]
    rrf_recalls = rrf_model[1].split(",")
    last_recall_rrf = rrf_recalls[-1]
    b_pref_rrf = float(rrf_model[3])

    rrf_results[0].append(float(last_precision_rrf))
    rrf_results[1].append(float(last_recall_rrf))
    rrf_results[2].append(b_pref_rrf)

    # Getting All CombSum Results.
    # Written in file with the following format: (Precisions, Recalls, Docs_Ids, B_PREF)
    comb_model = lines[5].replace("(", "").replace(")", "").replace("[", "").split("],")

    comb_precisions = comb_model[0].split(",")
    last_precision_comb = comb_precisions[-1]
    comb_recalls = comb_model[1].split(",")
    last_recall_comb = comb_recalls[-1]
    b_pref_comb = float(comb_model[3])

    comb_results[0].append(float(last_precision_comb))
    comb_results[1].append(float(last_recall_comb))
    comb_results[2].append(b_pref_comb)

    # Getting All CombSum Results.
    # Written in file with the following format: (Precisions, Recalls, Docs_Ids, B_PREF)
    comb_mnz_model = lines[6].replace("(", "").replace(")", "").replace("[", "").split("],")

    comb_mnz_precisions = comb_mnz_model[0].split(",")
    last_precision_comb_mnz = comb_mnz_precisions[-1]
    comb_mnz_recalls = comb_mnz_model[1].split(",")
    last_recall_comb_mnz = comb_mnz_recalls[-1]
    b_pref_comb_mnz = float(comb_mnz_model[3])

    comb_mnz_results[0].append(float(last_precision_comb_mnz))
    comb_mnz_results[1].append(float(last_recall_comb_mnz))
    comb_mnz_results[2].append(b_pref_comb_mnz)
    
    # Getting Average Models Precision Results.
    avg = (boolean_precision + float(last_precision_svm) + float(last_precision_bm)) / 3
    avg_precisions_topics.append(avg)

''' #THIS WAS FOR THE TABLE WHERE WE EXAMINE THE DIFF BETWEEN TOPICS 
if not os.path.isfile("./Results/TopicsPrecisions_Exp.csv"):
    data_frame = pandas.DataFrame([avg_precisions_topics], columns=topics, index=[results_folder])
    data_frame.to_csv("./Results/TopicsPrecisions_Exp.csv")
else:
    data_frame = pandas.read_csv("./Results/TopicsPrecisions_Exp.csv", index_col=0)
    data_frame_next = pandas.DataFrame([avg_precisions_topics], columns=topics, index=[results_folder])
    print(data_frame.head())
    data_frame = data_frame.append(data_frame_next)
    data_frame.to_csv("./Results/TopicsPrecisions_Exp.csv")
    print(data_frame.head())'''

# SAVING THE MODELS PRECISION AVERAGE (OF ALL TOPICS) FOR A CERTAIN EXPERIENCE
boolean_precision_topics = sum(boolean_model_results[0])/len(boolean_model_results[0])
boolean_recall_topics = sum(boolean_model_results[1])/len(boolean_model_results[1])
boolean_f_score_topics = sum(boolean_model_results[2])/len(boolean_model_results[2])
boolean_size_topics = sum(boolean_model_results[3])/len(boolean_model_results[3])

if not os.path.isfile("./Results/Boolean_Exp.csv"):
    data_frame = pandas.DataFrame([[boolean_precision_topics], [boolean_recall_topics], [boolean_f_score_topics]], columns=[5],
                                  index=['precision', 'recall', 'f-score'])
    data_frame.to_csv("./Results/Boolean_Exp.csv")
else:
    data_frame = pandas.read_csv("./Results/Boolean_Exp.csv", index_col=0)
    data_frame_next = pandas.DataFrame([[boolean_precision_topics], [boolean_recall_topics], [boolean_f_score_topics]], columns=[5],
                                       index=['precision', 'recall', 'f-score'])
    print(data_frame.head())
    data_frame[5] = [boolean_precision_topics, boolean_recall_topics, boolean_f_score_topics]
    data_frame.to_csv("./Results/Boolean_Exp.csv")
    print(data_frame.head())


svm_precision_topics = sum(space_vector_results[0])/len(space_vector_results[0])
bm25_precision_topics = sum(bm25_results[0])/len(bm25_results[0])
rrf_precision_topics = sum(rrf_results[0])/len(rrf_results[0])
comb_precision_topics = sum(comb_results[0])/len(comb_results[0])
comb_mnz_precision_topics = sum(comb_mnz_results[0])/len(comb_mnz_results[0])

if not os.path.isfile("./Results/ModelsPrecisions_Exp.csv"):
    data_frame = pandas.DataFrame([[boolean_precision_topics], [svm_precision_topics], [bm25_precision_topics],
                                   [rrf_precision_topics], [comb_precision_topics], [comb_mnz_precision_topics]], columns=[results_folder],
                                  index=['boolean model', 'space model', 'bm25', 'rrf', 'comb_sum', 'comb_mnz'])
    data_frame.to_csv("./Results/ModelsPrecisions_Exp.csv")
else:
    data_frame = pandas.read_csv("./Results/ModelsPrecisions_Exp.csv", index_col=0)
    data_frame_next = pandas.DataFrame([[boolean_precision_topics], [svm_precision_topics], [bm25_precision_topics],
                                        [rrf_precision_topics], [comb_precision_topics], [comb_mnz_precision_topics]], columns=[results_folder],
                                       index=['boolean model', 'space model', 'bm25', 'rrf',  'comb_sum', 'comb_mnz'])
    print(data_frame.head())
    data_frame[results_folder] = [boolean_precision_topics, svm_precision_topics, bm25_precision_topics,
                                  rrf_precision_topics, comb_precision_topics, comb_mnz_precision_topics]
    data_frame.to_csv("./Results/ModelsPrecisions_Exp.csv")
    print(data_frame.head())

####################################
#              Plots               #
####################################

# Plotting All Precision Results
fig = plt.subplots()
plt.figure(figsize=(50, 10))
plt.title('Precision of all models for all topics')
plt.plot(topics, space_vector_results[0], color='red', label="PrecisionSVM", markersize=20)
plt.plot(topics, boolean_model_results[0], color='blue', label="PrecisionBoolean", markersize=18)
plt.plot(topics, bm25_results[0], color='green', label="PrecisionBM25", markersize=16)
plt.plot(topics, rrf_results[0], color='yellow', label="PrecisionRRF", markersize=14)
plt.plot(topics, comb_results[0], color='black', label="PrecisionCombSum", markersize=12)
plt.plot(topics, comb_mnz_results[0], color='grey', label="PrecisionCombMNZ", markersize=10)
plt.legend()
plt.xlabel("Topics")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.savefig(folder_name + "plotTopics_Precision_Results.png")
plt.close()

# Plotting All Boolean Model Results
fig = plt.subplots()
plt.figure(figsize=(50, 10))
plt.title('Precision, Recall and F-Measure Of The Boolean Model for all topics')
plt.plot(topics, boolean_model_results[0], 'ro', color='blue', label="Precision")
plt.plot(topics, space_vector_results[1], 'ro', color='red', label="Recall")
plt.plot(topics, boolean_model_results[2], 'ro', color='green', label="F-Measure")
plt.legend()
plt.xlabel("Topics")
plt.ylabel("Boolean Model Results")
plt.ylim(0, 1)
plt.savefig(folder_name + "plotTopics_Boolean_Results.png")
plt.close()

# Plotting All Space Vector Model Results
fig = plt.subplots()
plt.figure(figsize=(50, 10))
plt.title('Average Precision, Average Recall and BPREF Of The Space Vector Model for all topics')
plt.plot(topics, space_vector_results[0], 'ro', color='blue', label="Average Precision")
plt.plot(topics, space_vector_results[1], 'ro', color='red', label="Average Recall")
plt.plot(topics, space_vector_results[2], 'ro', color='green', label="BPREF")
plt.legend()
plt.xlabel("Topics")
plt.ylabel("Space Vector Model Results")
plt.ylim(0, 1)
plt.savefig(folder_name + "plotTopics_SVM_Results.png")
plt.close()


# Plotting All BM25 Results
fig = plt.subplots()
plt.figure(figsize=(50, 10))
plt.title('Average Precision, Average Recall and BPREF Of BM25 for all topics')
plt.plot(topics, bm25_results[0], 'ro', color='blue', label="Average Precision")
plt.plot(topics, bm25_results[1], 'ro', color='red', label="Average Recall")
plt.plot(topics, bm25_results[2], 'ro', color='green', label="BPREF")
plt.legend()
plt.xlabel("Topics")
plt.ylabel("BM25 Results")
plt.ylim(0, 1)
plt.savefig(folder_name + "plotTopics_BM25_Results.png")
plt.close()

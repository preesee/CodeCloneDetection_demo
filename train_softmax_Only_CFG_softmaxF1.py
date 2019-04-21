__author__ = 'unique'
from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import itertools
import datetime
import random
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, Dense
from keras.optimizers import Adadelta
import csv
import pandas as pd
from ast import literal_eval
import os
from lib.data.Tree import *
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras import backend as K
from keras.activations import softmax
import keras as krs
from sklearn.metrics import classification_report
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.models import load_model
import datetime
import pickle
# TRAIN_CSV = '..../train.csv'

TRAIN_CSV = 'C:\\work\\codeclone_data\\preprocessDataFromDB\\CLONE_PAIRS_FULL.csv'
TRAIN_PAIRS_JSON_FILE = "C:\\work\\codeclone_data\\TRAIN_PAIRS_JSON_FILE.json"
# TEST_CSV = '..../test.csv'
CFG_EMBEDDING_FILE = 'C:\\work\\codeclone_data\\code_clone_embedding_for_CFG_model.txt'
SOURCE_CODE_EMBEDDING_FILE = 'C:\\work\\codeclone_data\\train_xe.java.code.gz'
MODEL_SAVING_DIR = 'C:\\work\\codeclone_data\\save_model\\'
is_dot_file = 'is_dot_file'
target_java_named_folder = 'target_java_named_folder'
target_java_file = 'target_java_file'
dot_file_list = 'dot_file_list'
bcb_reduced_dotfiles = 'bcb_reduced_dotfiles'
bcb_reduced_java_named_files = 'bcb_reduced_java_named_files'

code_dot_data_cfg_generated = "C:\\work\\codeclone_data\\preprocessDataFromDB\\code_dot_data_cfg_generated.csv"
clone_pairs_file = "C:\\work\\codeclone_data\\preprocessDataFromDB\\CLONE_PAIRS_FULL.csv"
save_source_code_file = 'save_source_java_code_txt_data.txt'
#MODEL_SAVING_DIR='/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/result/models/'



#TRAIN_CSV = '/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/CLONE_PAIRS_FULL.csv'
#TRAIN_PAIRS_JSON_FILE = "/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/TRAIN_PAIRS_JSON_FILE.json"
## TEST_CSV = '..../test.csv'
#CFG_EMBEDDING_FILE = '/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/code_clone_embedding_for_CFG_model.txt'
#SOURCE_CODE_EMBEDDING_FILE = '/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/train_xe.java.code.gz'
##MODEL_SAVING_DIR = 'C:\\work\\codeclone_data\\save_model\\'
#FIG_SAVING_DIR='/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/result/figures/'
#MODEL_SAVING_DIR='/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/result/models/'
#is_dot_file = 'is_dot_file'
#target_java_named_folder = 'target_java_named_folder'
#target_java_file = 'target_java_file'
#dot_file_list = 'dot_file_list'
#bcb_reduced_dotfiles = 'bcb_reduced_dotfiles'
#bcb_reduced_java_named_files = 'bcb_reduced_java_named_files'

#code_dot_data_cfg_generated = "C:\\work\\codeclone_data\\preprocessDataFromDB\\code_dot_data_cfg_generated.csv"
#clone_pairs_file = "C:\\work\\codeclone_data\\preprocessDataFromDB\\CLONE_PAIRS_FULL.csv"
#save_source_code_file = 'save_source_java_code_txt_data.txt'

def classifaction_report_csv(report, csvfile):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(csvfile, index = False)



def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2* ((precision * recall) / (precision + recall + K.epsilon()))


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data)), axis=1)
        val_targ = np.argmax(self.validation_data, axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        print(' — val_f1:' ,_val_f1)
        return


def read_source_code(source_file, startline, endline):
    f = source_file  # os.path.join(source_file_path, source_file)
    # source_plain_text = ''
    fh = open(f, "r", encoding='utf-8')
    source_code_txt = fh.readlines()[startline:endline]
    if len(source_code_txt) == 0:
        raise Exception("source_code_txt load =0")
    fh.close()
    dotfile_name = find_dotf_name(source_code_txt[0]).replace('(', '')
    source_plain_text = remove_comments(''.join(source_code_txt).replace("\n", " ").replace("\t", " "))
    return source_plain_text, dotfile_name


def remove_comments(text):
    """Remove C-style /*comments*/ from a string."""
    p = r'/\*[^*]*\*+([^/*][^*]*\*+)*/|("(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'|.[^/"\'\\]*)'
    return ''.join(m.group(2) for m in re.finditer(p, text, re.M | re.S) if m.group(2))


def find_dotf_name(dotf_name):
    pattern = ' \S*?\('
    dotf_name = re.search(pattern, dotf_name, flags=0)
    if dotf_name is not None:
        dotf_name = dotf_name.group(0).split(' ')[1]

    else:
        dotf_name = None
    return dotf_name


def create_CFG_structor(clone_data, dotfile_data_dict):
    try:

        # global data, source_plain_text, dotfile_name, dotfilespath, dotfname, fulldotfilepath
        data_tuple = literal_eval(clone_data)
        data = dotfile_data_dict[data_tuple[0]]  # find in dict  ('77069.java', 872, 898)
        source_plain_text, dotfile_name = read_source_code(data[2], data_tuple[1] - 1, data_tuple[2])
        dotfilespath = data[1].replace(bcb_reduced_java_named_files, bcb_reduced_dotfiles)
        dotfname = data[3][dotfile_name]  # find dotfile path
        fulldotfilepath = os.path.join(dotfilespath, dotfname)
        # cfg = pydot.graph_from_dot_file(fulldotfilepath)
        cfg_idx = 'g_' + data[0].split('.')[0] + "_" + dotfname.replace('.dot', '')
        return cfg_idx, source_plain_text
    except Exception as e:
        print(e)


def write2file(save_file_path, write_str):
    with open(save_file_path, 'a') as f:
        f.write("%s\n" % write_str)


# load code_dot_data as dict with dot file information and source code
dotfile_data_dict = {}


# create data source plain txt lib
def create_java_source_code_data():
    with open(code_dot_data_cfg_generated, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        next(csv_reader)  # skip header
        for item in csv_reader:
            plant_source_code_text = read_source_code(item[target_java_file])
            write2file(save_source_code_file, plant_source_code_text.encode("utf-8", 'ignore'))


# create_java_source_code_data()
doc2vec_cfg = Doc2Vec.load(CFG_EMBEDDING_FILE)

word2vec = KeyedVectors.load(SOURCE_CODE_EMBEDDING_FILE)
# with open(code_dot_data_cfg_generated, mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     next(csv_reader)  # skip header
#     for item in csv_reader:
#         dot_list = item[dot_file_list].split('|')
#         dotfiledict = {}
#         for dfname in dot_list:
#             dot_fname = find_dotf_name(dfname).replace('(', '')
#             dotfiledict[dot_fname] = dfname
#         row = (
#             item['file_name'],
#             item['target_java_named_folder'],
#             item['target_java_named_folder'].replace('bcb_reduced_java_named_files', 'bcb_reduced') + '.java',
#             dotfiledict)
#         dotfile_data_dict[item['file_name']] = row

    # data = [r for r in csv_reader]

# data = pd.read_csv(code_dot_data_cfg_generated).to_dict(orient="row")

# train_df = pd.read_csv(TRAIN_CSV)
# for dataset in [train_df]:
#     for index, row in dataset.iterrows():
#         row['code_clone1']
# load clone_pair information with text and source files information


#train_data = []  # clone_pair,
# rebuild_json_cfg = False
# if rebuild_json_cfg:
#     with open(clone_pairs_file, mode='r') as csv_file:  # data pair source code
#         csv_reader = csv.DictReader(csv_file)
#         next(csv_reader)  # skip header
#         items = []
#         for item in csv_reader:
#
#             # format file name , start line , end line
#             # data_l = literal_eval(item['CLONE_NAME1'])
#             # data = dotfile_data_dict[data_l[0]]
#             # source_plain_text, dotfile_name = read_source_code(data[2], data_l[1] - 1, data_l[2])
#             # dotfilespath = data[1].replace(bcb_reduced_java_named_files, bcb_reduced_dotfiles)
#             # dotfname = data[3][dotfile_name]  # find dotfile path
#             # fulldotfilepath = os.path.join(dotfilespath, dotfname)
#             # dot_a = pydot.graph_from_dot_file(fulldotfilepath)
#             try:
#                 data_dic = {}
#                 cfg_idx_a, source_plain_text_a = create_CFG_structor(item['CLONE_NAME1'], dotfile_data_dict)
#                 cfg_idx_b, source_plain_text_b = create_CFG_structor(item['CLONE_NAME2'], dotfile_data_dict)
#                 data_dic['code_clone1'] = source_plain_text_a
#                 data_dic['code_clone2'] = source_plain_text_b
#                 data_dic['cfg_idx_A'] = cfg_idx_a
#                 data_dic['cfg_idx_B'] = cfg_idx_b
#                 data_dic['TYPE'] = item['TYPE']
#                 if cfg_idx_a in doc2vec_cfg.docvecs:
#                     pass
#                 else:
#                     continue
#                 if cfg_idx_b in doc2vec_cfg.docvecs:
#                     train_data.append(data_dic)
#                 else:
#                     continue
#
#
#             except Exception as e:
#                 print(e)
#                 continue
#
#             print("done")
#
#     with open(TRAIN_PAIRS_JSON_FILE, 'w') as f:
#         json.dump(train_data, f, indent=1)
label_data_convertor = {}
label_data_convertor['T4'] = 4
label_data_convertor['T2'] = 1
label_data_convertor['T1'] = 0
label_data_convertor['MT3'] = 2
label_data_convertor['ST3'] = 3
data_loaded = json.load(open(TRAIN_PAIRS_JSON_FILE))

datarowsT4 = [row for row in data_loaded if row['TYPE'] == 'T4']
datarowsT1 = [row for row in data_loaded if row['TYPE'] == 'T1']
datarowsT2 = [row for row in data_loaded if row['TYPE'] == 'T2']
datarowsMT3 = [row for row in data_loaded if row['TYPE'] == 'MT3']
datarowsST3 = [row for row in data_loaded if row['TYPE'] == 'ST3']
datafortrain = []
datafortrain += datarowsT4[:18000]
datafortrain += datarowsT1[:]
datafortrain += datarowsT2[:]
datafortrain += datarowsMT3[:]
datafortrain += datarowsST3[:]

# Load training and test set

# test_df = ''  # pd.read_csv(TEST_CSV)

stops = set(stopwords.words('english'))


def source_code_to_tokens(text):
    tokens = java_tokenize(text)
    return tokens


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"COMMA", " ", text)
    text = re.sub(r"RETURN_BACK", " ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\[]", " ", text)
    text = re.sub(r"\[", " ", text)
    text = re.sub(r"\]", " ", text)
    text = re.sub(r"\%", " % ", text)
    text = re.sub(r">", " > ", text)
    text = re.sub(r"<", " < ", text)
    # text = re.sub(r"\!\=", " != ", text)
    text = re.sub(r"()", " ", text)

    text = text.split()

    return text


samples_number = 500

validation_percent = 0.2
test_percent = 0.2
r_data_loaded = random.sample(datafortrain, len(datafortrain))
r_samples = r_data_loaded[:samples_number]
df = pd.DataFrame(r_samples)
train_df = df


#test_numbers=4000
# r_test_samples = r_data_loaded[16000:20000]
# df_test = pd.DataFrame(r_test_samples)
# test_df = df
# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

# gensim.models.Word2Vec.load_word2vec_format('/data5/momo-projects/user_interest_classification/code/word2vec/vectors_groups_1105.bin', binary=True, unicode_errors='ignore')
code_clones_cols = ['code_clone1', 'code_clone2']

# Iterate over the questions only of both training and test datasets
for dataset in [train_df]:
    for index, row in dataset.iterrows():
        print("dataset rows:" + str(index / len(dataset)) + "%")
        # Iterate through the text of both questions of the row
        for code_clone in code_clones_cols:

            q2n = []  # q2n -> question numbers representation
            tokens_for_sents = source_code_to_tokens(row[code_clone])

            for word in tokens_for_sents:

                # Check for unwanted words
                if word in stops and word not in word2vec.wv.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.set_value(index, code_clone, q2n)

embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.wv.vocab:
        embeddings[index] = word2vec.wv.word_vec(word)

del word2vec

# doc to vec embedding preparation
#cfg_embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
#cfg_embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
docslen=(len(doc2vec_cfg.docvecs.doctags))
print(docslen)

cfg_embedding_matrix =1* np.random.randn(len(doc2vec_cfg.docvecs.doctags) + 1, embedding_dim)
cfg_doc_dict={}
i = 0
for doc_tag in doc2vec_cfg.docvecs.doctags:
    cfg_doc_dict[doc_tag]=i+1
    cfg_embedding_matrix[i + 1] = doc2vec_cfg.docvecs[doc_tag]
    i = i + 1

max_seq_length = max(train_df.code_clone1.map(lambda x: len(x)).max(),
                     train_df.code_clone2.map(lambda x: len(x)).max(),)
                      # test_df.code_clone1.map(lambda x: len(x)).max(),
                      # test_df.code_clone2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = int(len(train_df) * validation_percent)
training_size = len(train_df) - validation_size

X = train_df[code_clones_cols]
Y = train_df.TYPE.map(lambda x: label_data_convertor[x])
#Y_test = test_df.TYPE.map(lambda x: label_data_convertor[x])

cfg_pair = {'cfg_left': train_df.cfg_idx_A, 'cfg_right': train_df.cfg_idx_B}
X_train, X_validation, cfg_idx_A_train, cfg_idx_A_validation, cfg_idx_B_train, cfg_idx_B_validation, Y_train, Y_validation = train_test_split(
    X, train_df.cfg_idx_A, train_df.cfg_idx_B, Y, test_size=validation_size)

X_train, X_test, cfg_idx_A_train, cfg_idx_A_test, cfg_idx_B_train, cfg_idx_B_test, Y_train, Y_test = train_test_split(
    X, train_df.cfg_idx_A, train_df.cfg_idx_B, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.code_clone1, 'right': X_train.code_clone2,
           'cfg_A': pd.Series([cfg_doc_dict[cfg] for cfg in cfg_idx_A_train.values]),  #'cfg_A': [doc2vec_cfg.docvecs[cfg] for cfg in cfg_idx_A_train],
           #'cfg_A': cfg_idx_A_train,
           'cfg_B': pd.Series( [cfg_doc_dict[cfg] for cfg in cfg_idx_B_train.values])}# 'cfg_A': [doc2vec_cfg.docvecs[cfg] for cfg in cfg_idx_A_train],
           #'cfg_B': cfg_idx_B_train}
X_validation = {'left': X_validation.code_clone1, 'right': X_validation.code_clone2,
                'cfg_A':  pd.Series([cfg_doc_dict[cfg] for cfg in cfg_idx_A_validation.values]),
                'cfg_B':  pd.Series([cfg_doc_dict[cfg] for cfg in cfg_idx_B_validation.values])}
X_test = {'left': X_test.code_clone1, 'right': X_test.code_clone2,
                'cfg_A':  pd.Series([cfg_doc_dict[cfg] for cfg in cfg_idx_A_test.values]),
                'cfg_B':  pd.Series([cfg_doc_dict[cfg] for cfg in cfg_idx_B_test.values])}

# X_test = {'left': test_df.code_clone1, 'right': test_df.code_clone2,
#                 'cfg_A':  pd.Series([cfg_doc_dict[cfg] for cfg in test_df.cfg_idx_A.values]),
#                 # 'cfg_B': [doc2vec_cfg.docvecs[cfg] for cfg in cfg_idx_B_train]}
#                 'cfg_B':  pd.Series([cfg_doc_dict[cfg] for cfg in test_df.cfg_idx_B.values])}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values
#Y_test=Y_test.values
# Zero padding
for dataset, side in itertools.product([X_train, X_validation,X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# Model variables
n_hidden = 128
gradient_clipping_norm = 1.25
batch_size = 32
n_epoch = 50


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def classification_softmax(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.sum(K.abs(left - right), axis=1, keepdims=True)


def softMaxAxis1(x):
    return softmax(x, axis=1)


# The visible layer
#left_input = Input(shape=(max_seq_length,), dtype='int32')
#right_input = Input(shape=(max_seq_length,), dtype='int32')
#cfg_left_input = Input(shape=(max_seq_length,))
cfg_left_input = Input(shape=(1,),dtype='int32')
#cfg_right_input = Input(shape=(max_seq_length,))
cfg_right_input = Input(shape=(1,),dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length,
                            trainable=False)

#https://stackoverflow.com/questions/48999199/export-gensim-doc2vec-embeddings-into-separate-file-to-use-with-keras-embedding can be replace with embedding lay

embedding_cfg_layer =  Embedding(len(cfg_embedding_matrix), embedding_dim, weights=[cfg_embedding_matrix], input_length=1,
                            trainable=False)

# Embedded version of the inputs
# sess = K.get_session()
# array_l = sess.run(cfg_left_input)
# array_r = sess.run(cfg_right_input)
# cfg_left=doc2vec_cfg[array_l[0]]
# cfg_right=doc2vec_cfg[array_r[0]]
# cfg_left = doc2vec_cfg.docvecs[cfg_left_input]
# cfg_right = doc2vec_cfg.docvecs[cfg_right_input]


cfg_embedding_l=embedding_cfg_layer(cfg_left_input)
cfg_embedding_r=embedding_cfg_layer(cfg_right_input)
#encoded_left = krs.layers.Concatenate(axis=1)([embedding_layer(left_input),cfg_embedding_l])
#encoded_right = krs.layers.Concatenate(axis=1)([embedding_layer(right_input), cfg_embedding_r])

encoded_left = cfg_embedding_l#embedding_layer(left_input)
encoded_right = cfg_embedding_r#embedding_layer(right_input)
# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                         output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

dist = Lambda(lambda x: classification_softmax(x[0], x[1]))([left_output, right_output])
classify = Dense(5, activation="softmax")(dist)
# Pack it all up into a model
malstm = Model([cfg_left_input,cfg_right_input], [classify])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

# malstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', f1, recall,precision])
malstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])#, f1, recall, precision])

# Start training
training_start_time = time()

metrics = Metrics()
malstm_trained = malstm.fit(
    [X_train['cfg_A'], X_train['cfg_B']],
    krs.utils.to_categorical(Y_train, 5),
    batch_size=batch_size, nb_epoch=n_epoch,
    #callbacks=[metrics],
    validation_data=(
        [ X_validation['cfg_A'],X_validation['cfg_B']],
        krs.utils.to_categorical(Y_validation, 5)))

print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                        datetime.timedelta(seconds=time() - training_start_time)))
scores = malstm.evaluate([ X_test['cfg_A'],
                          X_test['cfg_B']], krs.utils.to_categorical(Y_test, 5))
print("%s: %.2f%%" % (malstm.metrics_names[1], scores[1] * 100))
for i in range(len(malstm.metrics_names)):
    print("%s: %.2f%%" % (malstm.metrics_names[i], scores[i]))

y_pred = malstm.predict( [X_test['cfg_A'],X_test['cfg_B']])
report=classification_report(Y_test, np.argmax(y_pred,1))
print(report)
csvfile=MODEL_SAVING_DIR+'classification_report_only_cfg.csv'
classifaction_report_csv(report, csvfile)
# Plot accuracy
# plt.plot(malstm_trained.history['categorical_accuracy'])
# plt.plot(malstm_trained.history['val_categorical_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
#
# plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc.png')
# #plt.show()
#
# # Plot loss
# plt.plot(malstm_trained.history['loss'])
# plt.plot(malstm_trained.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'loss.png')
#plt.show()

# Plot of F1
# plt.plot(malstm_trained.history['f1'])
# plt.plot(malstm_trained.history['val_f1'])
# plt.title('F1')
# plt.ylabel('F1')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'F1.png')
# plt.show()
#
# # Plot of Precision
# plt.plot(malstm_trained.history['precision'])
# plt.plot(malstm_trained.history['val_precision'])
# plt.title('Precision')
# plt.ylabel('Precision')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'Precision.png')
# plt.show()
#
# # Plot of recall
# plt.plot(malstm_trained.history['recall'])
# plt.plot(malstm_trained.history['val_recall'])
# plt.title('recall')
# plt.ylabel('recall')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'Recall.png')
#plt.show()


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'loss_cfg_only.png')
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc_cfg_only.png')
    plt.show()


plot_history(malstm_trained)

malstm_trained_json=malstm_trained.model.to_json()
with open(MODEL_SAVING_DIR+str(datetime.datetime.now())+'loss_cfg_only.model.json',"w") as json_file:
    json_file.write(malstm_trained_json)
malstm_trained.model.save_weights(MODEL_SAVING_DIR+str(datetime.datetime.now())+'loss_cfg_only.model.h5')
print("save model to disk done")

#load json and create model
# json_loaded=open(modelname,'r')
# loaded_model_json=json_loaded.read()
# json_loaded.close()
# loaded_model=model_from_json(loaded_model_json)
# loaded_model.load_weights(modelwightname)

hist=malstm_trained.history
modelname=MODEL_SAVING_DIR+str(datetime.datetime.now())+'loss_cfg_only.model.pkl'

with open(modelname,"wb") as file_pi:
    pickle.dump(hist,file_pi)
print("model save done!")
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
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
import csv
import pandas as pd
import re
from ast import literal_eval
import os
from lib.data.Tree import *
#import pydot
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras import backend as K
# TRAIN_CSV = '..../train.csv'

TRAIN_CSV = '/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/CLONE_PAIRS_FULL.csv'
TRAIN_PAIRS_JSON_FILE = "/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/TRAIN_PAIRS_JSON_FILE.json"
# TEST_CSV = '..../test.csv'
CFG_EMBEDDING_FILE = '/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/code_clone_embedding_for_CFG_model.txt'
SOURCE_CODE_EMBEDDING_FILE = '/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/train_xe.java.code.gz'
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
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict(self.validation_data[0])

        val_predict = [1 if x[0] > 0.47 else 0 for x in val_predict]
        val_targ = self.validation_data[1]

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('— val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))
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


# def read_source_code(source_file):
#     f = source_file  # os.path.join(source_file_path, source_file)
#     # source_plain_text = ''
#     fh = open(f, "r", encoding='utf-8')
#     source_code_txt = fh.readlines()[:]
#     fh.close()
#     source_plain_text = remove_comments(''.join(source_code_txt).replace("\n", " ").replace("\t", " "))
#     return source_plain_text


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

# train_data = []  # clone_pair, cfg_idx_pair
# with open(clone_pairs_file, mode='r') as csv_file:  # data pair source code
#     csv_reader = csv.DictReader(csv_file)
#     next(csv_reader)  # skip header
#     items=[]
#     for item in csv_reader:
#
#         # format file name , start line , end line
#         # data_l = literal_eval(item['CLONE_NAME1'])
#         # data = dotfile_data_dict[data_l[0]]
#         # source_plain_text, dotfile_name = read_source_code(data[2], data_l[1] - 1, data_l[2])
#         # dotfilespath = data[1].replace(bcb_reduced_java_named_files, bcb_reduced_dotfiles)
#         # dotfname = data[3][dotfile_name]  # find dotfile path
#         # fulldotfilepath = os.path.join(dotfilespath, dotfname)
#         # dot_a = pydot.graph_from_dot_file(fulldotfilepath)
#         try:
#             data_dic={}
#             cfg_idx_a, source_plain_text_a = create_CFG_structor(item['CLONE_NAME1'], dotfile_data_dict)
#             cfg_idx_b, source_plain_text_b = create_CFG_structor(item['CLONE_NAME2'], dotfile_data_dict)
#             data_dic['code_clone1']=source_plain_text_a
#             data_dic['code_clone2'] = source_plain_text_b
#             data_dic['cfg_idx_A'] = cfg_idx_a
#             data_dic['cfg_idx_B'] = cfg_idx_b
#             data_dic['TYPE']=item['TYPE']
#             train_data.append(data_dic)
#         except Exception as e:
#             print(e)
#             continue
#
#         print("done")
#
# with open(TRAIN_PAIRS_JSON_FILE, 'w') as f:
#     json.dump(train_data, f, indent=1)
label_data_convertor = {}
label_data_convertor['T4'] = 1
label_data_convertor['T2'] = 0
label_data_convertor['T1'] = 0
label_data_convertor['MT3'] = 0
label_data_convertor['ST3'] = 0
data_loaded = json.load(open(TRAIN_PAIRS_JSON_FILE))

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


r_data_loaded = random.sample(data_loaded, len(data_loaded))
r_samples = r_data_loaded[:100000]
df = pd.DataFrame(r_samples)
train_df = df
# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
word2vec_cfg = Doc2Vec.load(CFG_EMBEDDING_FILE)
word2vec = KeyedVectors.load(SOURCE_CODE_EMBEDDING_FILE)
# gensim.models.Word2Vec.load_word2vec_format('/data5/momo-projects/user_interest_classification/code/word2vec/vectors_groups_1105.bin', binary=True, unicode_errors='ignore')
code_clones_cols = ['code_clone1', 'code_clone2']

# Iterate over the questions only of both training and test datasets
for dataset in [train_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for code_clone in code_clones_cols:

            q2n = []  # q2n -> question numbers representation
            for word in source_code_to_tokens(row[code_clone]):

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

max_seq_length = max(train_df.code_clone1.map(lambda x: len(x)).max(),
                     train_df.code_clone2.map(lambda x: len(x)).max(), )
# test_df.code_clone1.map(lambda x: len(x)).max(),
# test_df.code_clone2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = int(len(train_df)*0.2)
training_size = len(train_df) - validation_size

X = train_df[code_clones_cols]
Y = train_df.TYPE.map(lambda x: label_data_convertor[x])

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.code_clone1, 'right': X_train.code_clone2}
X_validation = {'left': X_validation.code_clone1, 'right': X_validation.code_clone2}
# X_test = {'left': test_df.code_clone1, 'right': test_df.code_clone2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 15


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length,
                            trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                         output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', f1, recall,precision])

# Start training
training_start_time = time()
metrics = Metrics()
malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            # callbacks=[metrics],
                            verbose=1,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                        datetime.timedelta(seconds=time() - training_start_time)))
scores = malstm.evaluate([X_validation['left'], X_validation['right']], Y_validation)
print("%s: %.2f%%" % (malstm.metrics_names[1], scores[1]*100))

# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

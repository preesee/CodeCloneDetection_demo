from keras import Input
import keras
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from keras.layers import *
from keras.layers.core import Dense, Dropout
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
import pandas as pd
from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from IPython.display import HTML, display
import seaborn as sns
import pandas as pd
import numpy as np
#from keras.preprocessing import sequences_to_texts
# https://blog.csdn.net/uhauha2929/article/details/80733255 blog version  data, code on kaggle: https://www.kaggle.com/c/word2vec-nlp-tutorial/data
# https://blog.csdn.net/fkyyly/article/details/82501126 tensor flow version
# https://blog.csdn.net/yanhe156/article/details/85476608 keras version without hierachical attention  code https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-672-lb/notebook
#https://stackoverflow.com/questions/53867351/how-to-visualize-attention-weights vis attention
# https://github.com/uhauha2929/examples/blob/master/self-attention.ipynb  self attention
#https://www.jianshu.com/p/1e2a63cc9ba3 pytorch attention vis
#https://python-graph-gallery.com/92-control-color-in-seaborn-heatmaps/  heatmaps 
#https://stackoverflow.com/questions/47585775/seaborn-heatmap-with-single-column
#https://stackoverflow.com/questions/43330205/heatmap-from-columns-in-pandas-dataframe/43333447
#https://www.quantinsti.com/blog/creating-heatmap-using-python-seaborn
#https://stackoverflow.com/questions/41971587/how-to-convert-predicted-sequence-back-to-text-in-keras
#https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb 
_han = False




# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(seq, attns):
    html = ""
    for ix, attn in zip(seq, attns):
        html += ' ' + highlight(
            TEXT.vocab.itos[ix],
            attn
        )
    return html + "<br><br>\n"
class CharVal(object):
    def __init__(self, char, val):
        char=   list(map(sequence_to_text, char))
        self.char = ' '.join([f'{e}' for e in char[0]]).replace('None','')
        self.val = val

    def __str__(self):
        return self.char

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
def color_charvals(s):
    r = 255-int(s.val*255)
    color = rgb_to_hex((255, r, r))
    return 'background-color: %s' % color

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        # general
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

data_train = pd.read_csv('.\\train_data\\labeledTrainData.tsv', sep='\t')
print(data_train.shape)

reviews = []
labels = []
texts = []
rows=data_train.review.shape[0]
#rows=250

for idx in range(rows):
    text = data_train.review[idx]
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)

    labels.append(data_train.sentiment[idx])

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SENT_LENGTH)
if _han == True:
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
    preds = Dense(2, activation='softmax')(l_lstm_sent)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting - Hierachical LSTM")


if _han == False:
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
   # l_dense= l_lstm
    l_att = AttentionLayer(name='attention_vec')(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    #l_dense_sent= l_lstm_sent
    l_att_sent = AttentionLayer()(l_dense_sent)
    preds = Dense(2, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)
    #model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting - Hierachical attention network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=10, batch_size=50)

model = Model(inputs=model.input,
              outputs=[model.output, model.get_layer('attention_layer_1').output])

gooutputs = model.predict(x_val)
model_outputs = outputs[0]
attention_outputs = outputs[1]

# if you are using batches the outputs will be in batches
# get exact attentions of chars

#reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

an_attention_output = attention_outputs[0][-len(x_val):]

# before the prediction i supposed you tokenized text
# you need to match each char and attention
sent_number=1
char_vals = [CharVal(c, v) for c, v in zip(x_val, an_attention_output)]
import pandas as pd
df = pd.DataFrame(char_vals).transpose()
# apply coloring values
#char_df = char_df.style.applymap(color_charvals).to_html()
sns.heatmap(df, cmap="YlGnBu")
sns.heatmap(df, cmap="Blues")
sns.heatmap(df, cmap="BuPu")
sns.heatmap(df, cmap="Greens")
sns.plt.show()

with open(r'c:\tmp\html.html', 'w') as html:
    html.write(char_df.render())
char_df.to_html(r'c:\temp\html.html')

with pd.option_context('display.precision', 2):
    html = (df.style
              .applymap(color_negative_red)
              .apply(highlight_max))
html

model_save_name='model.save'
MODEL_SAVING_DIR="C:\\work\\current_codeClone\\"
model_trained_json=model.to_json()
model_save_path=MODEL_SAVING_DIR+model_save_name+'.json'
with open(modelname,"w") as json_file:
    json_file.write(model_trained_json)
model_wight_name=MODEL_SAVING_DIR+model_save_name+'.h5'
malstm_trained.model.save_weights(model_wight_name)
print("savemodel to disk")

# load json and create model
# json_loaded=open(modelname,'r')
# loaded_model_json=json_loaded.read()
# json_loaded.close()
# loaded_model=model_from_json(loaded_model_json)
# loaded_model.load_weights(modelwightname)
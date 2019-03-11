
__author__ = 'unique'

import matplotlib.pyplot as plt

from keras.models import model_from_json
import numpy
import datetime
from keras.activations import softmax

from keras import backend as K
import pickle

FIG_SAVING_DIR='/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/result/figures/'
MODELS_SAVING_DIR='/home/wehua/PycharmProjects/code_clone/code_clone/code_clone_detections/codeclone_data/result/models/'
modelname=''

modelwightname=''


def classification_softmax(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.sum(K.abs(left - right), axis=1, keepdims=True)

def load_model(modelname,modelwightname):
    json_loaded = open(modelname, 'rb')
    loaded_model_json = json_loaded.read()
    json_loaded.close()
    loaded_model = model_from_json(loaded_model_json,custom_objects={"classification_softmax":classification_softmax})
    loaded_model.load_weights(modelwightname)
    return loaded_model

def load_history(modelname):

    with open(modelname, 'rb') as pick_file:
        loaded_hist = pickle.load(pick_file)

    return loaded_hist

#load json and create model


colors=['b','g','r','c','m','y','k','w']
def plot_history_train_loss(historys):


    plt.figure(1)
    for k,history in  historys.items():


        loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
        #loss_dic[k]=loss_list
        val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
        #val_loss_dic[k]=val_loss_list


        acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
        #acc_dic[k]=acc_list
        val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
        #val_acc_dic[k]=val_acc_list
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return

        ## As loss always exists
        epochs = range(1,len(history[loss_list[0]]) + 1)

        ## Loss

        for l in loss_list:
            plt.plot(epochs, history[l], colors[list(historys.keys()).index(k)], label=k)
        # for l in val_loss_list:
        #     plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(str(format(history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'loss_Multi.png')
        ## Accuracy
        # plt.figure(2)
        # for l in acc_list:
        #     plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
        # for l in val_acc_list:
        #     plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')
        #
        # plt.title('Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc_cfg_txt.png')
        # plt.show()


def plot_history_validate_loss(historys):
    # loss_dic={}
    # val_loss_dic={}
    # acc_dic={}
    # val_acc_dic={}
    plt.figure(2)
    for k,history in  historys.items():


        loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
        #loss_dic[k]=loss_list
        val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
        #val_loss_dic[k]=val_loss_list


        acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
        #acc_dic[k]=acc_list
        val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
        #val_acc_dic[k]=val_acc_list
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return

        ## As loss always exists
        epochs = range(1,len(history[loss_list[0]]) + 1)

        ## Loss

        # for l in loss_list:
        #     plt.plot(epochs, history[l], 'b', label=k)
        for l in val_loss_list:
            plt.plot(epochs, history[l], colors[list(historys.keys()).index(k)], label=k)

    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'validation_loss_Multi.png')
        ## Accuracy
        # plt.figure(2)
        # for l in acc_list:
        #     plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
        # for l in val_acc_list:
        #     plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')
        #
        # plt.title('Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc_cfg_txt.png')
        # plt.show()



def plot_history_train_acc(historys):
    # loss_dic={}
    # val_loss_dic={}
    # acc_dic={}
    # val_acc_dic={}
    plt.figure(3)
    for k,history in  historys.items():


        #loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
        #loss_dic[k]=loss_list
        #val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
        #val_loss_dic[k]=val_loss_list


        acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
        #acc_dic[k]=acc_list
        val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
        #val_acc_dic[k]=val_acc_list
        if len(acc_list) == 0:
            print('Loss is missing in history')
            return

        ## As loss always exists
        epochs = range(1,len(history[acc_list[0]]) + 1)

        ## Loss

        # for l in loss_list:
        #     plt.plot(epochs, history[l], 'b', label=k)
        for l in acc_list:
            plt.plot(epochs, history[l], colors[list(historys.keys()).index(k)], label=k)

    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc_Multi.png')
        ## Accuracy
        # plt.figure(2)
        # for l in acc_list:
        #     plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
        # for l in val_acc_list:
        #     plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')
        #
        # plt.title('Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc_cfg_txt.png')
        # plt.show()



def plot_history_validate_acc(historys):
    # loss_dic={}
    # val_loss_dic={}
    # acc_dic={}
    # val_acc_dic={}
    plt.figure(4)
    for k,history in  historys.items():


        #loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
        #loss_dic[k]=loss_list
        #val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
        #val_loss_dic[k]=val_loss_list


        #acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
        #acc_dic[k]=acc_list
        val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
        #val_acc_dic[k]=val_acc_list
        if len(val_acc_list) == 0:
            print('Loss is missing in history')
            return

        ## As loss always exists
        epochs = range(1,len(history[val_acc_list[0]]) + 1)

        ## Loss

        # for l in loss_list:
        #     plt.plot(epochs, history[l], 'b', label=k)
        for l in val_acc_list:
            plt.plot(epochs, history[l], colors[list(historys.keys()).index(k)], label=k)

    plt.title('valication accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'validation_acc_Multi.png')
        ## Accuracy
        # plt.figure(2)
        # for l in acc_list:
        #     plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
        # for l in val_acc_list:
        #     plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')
        #
        # plt.title('Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.savefig(FIG_SAVING_DIR+str(datetime.datetime.now())+'acc_cfg_txt.png')
        # plt.show()

loaded_hists={}


def softMaxAxis1(x):
    return softmax(x, axis=1)

modelname=MODELS_SAVING_DIR+'1_loss_cfg_only.model.pkl'
weightname=MODELS_SAVING_DIR+'1_loss_cfg_only.model.h5'
#load_history(modelname)
modelname=MODELS_SAVING_DIR+'1.pkl'
loaded_hists['CFG']=load_history(modelname)


modelname=MODELS_SAVING_DIR+'27_loss_cfg_only.model.pkl'
weightname=MODELS_SAVING_DIR+'27_loss_cfg_only.model.h5'
#load_model(modelname)
modelname=MODELS_SAVING_DIR+'2.pkl'
loaded_hists['AST']=load_history(modelname)#load_model(modelname,weightname)


modelname=MODELS_SAVING_DIR+'28_loss_cfg_only.model.pkl'
weightname=MODELS_SAVING_DIR+'28_loss_cfg_only.model.h5'
#load_model(modelname)
modelname=MODELS_SAVING_DIR+'3.pkl'
loaded_hists['TXT']=load_history(modelname)#load_model(modelname,weightname)

modelname=MODELS_SAVING_DIR+'29_loss_cfg_only.model.pkl'
weightname=MODELS_SAVING_DIR+'29_loss_cfg_only.model.h5'
#load_model(modelname)
modelname=MODELS_SAVING_DIR+'4.pkl'
loaded_hists['TXT + AST + CFG']=load_history(modelname)#load_model(modelname,weightname)


plot_history_train_loss(loaded_hists)
plot_history_train_acc(loaded_hists)
plot_history_validate_loss(loaded_hists)
plot_history_validate_acc(loaded_hists)
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras import optimizers,applications, callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import numpy as np
from wx_hyperparam import WxHyperParameter
#import xgboost as xgb

#set default global hyper paramerters

# yTrain need to one hot encoded
def NaiveSLPmodel(x_train, y_train, x_val, y_val, hyper_param):
    input_dim = len(x_train[0])
    output_dim = len(y_train[0])
    inputs = Input((input_dim,))
    #fc_out = Dense(2,  kernel_initializer='zeros', bias_initializer='zeros', activation='softmax')(inputs)
    fc_out = Dense(output_dim,  activation='softmax')(inputs)
    model = Model(inputs=inputs, outputs=fc_out)

    #build a optimizer
    sgd = optimizers.SGD(lr=hyper_param.learning_ratio, decay=hyper_param.weight_decay, momentum=hyper_param.momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])        

    #call backs
    def step_decay(epoch):
        exp_num = int(epoch/10)+1       
        return float(hyper_param.learning_ratio/(10 ** exp_num))

    best_model_path="checkpoint/slp_wx_weights_best"+".hdf5"
    save_best_model = ModelCheckpoint(best_model_path, monitor="val_loss", verbose=1, save_best_only=True, mode='min')  
    #save_best_model = ModelCheckpoint(best_model_path, monitor="val_acc", verbose=1, save_best_only=True, mode='max')
    change_lr = LearningRateScheduler( hyper_param.learning_ratio )                                

    #run train
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), 
                epochs=hyper_param.epochs, batch_size=hyper_param.batch_size, shuffle=True, callbacks=[save_best_model])

    #load best model
    model.load_weights(best_model_path)

    #load weights
    weights = model.get_weights()

    return model

def ClassifierLoocv(x_train, y_train, x_val, y_val, x_test, y_test):
    pass
    #clf = xgb.XGBClassifier(seed=1, objective='binary:logistic')
    #clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
    #pred_test = clf.predict(x_test)
    #return pred_test == y_test

def WxSlp(x_train, y_train, x_val, y_val, n_selection , hyper_param):#suppot 2 class classification only now.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    input_dim = len(x_train[0])
    num_cls = len(y_train[0])
    # make model and do train
    model = NaiveSLPmodel(x_train, y_train, x_val, y_val, hyper_param)

    #load weights
    weights = model.get_weights()

    #cacul WX scores
    num_data = {}
    running_avg={}
    tot_avg={}
    Wt = weights[0].transpose() #all weights of model
    Wb = weights[1].transpose() #all bias of model

    # Create Xk
    for i in range(num_cls): 
        tot_avg[i] = np.zeros(input_dim) # avg of input data for each output class
        num_data[i] = 0.
    for i in range(len(x_train)):
        c = y_train[i].argmax()
        x = x_train[i]
        tot_avg[c] = tot_avg[c] + x
        num_data[c] = num_data[c] + 1
    for i in range(num_cls):
        tot_avg[i] = tot_avg[i] / num_data[i]

    #data input for first class
    wx_00 = tot_avg[0] * Wt[0]# + Wb[0]# first class input avg * first class weight + first class bias
    wx_01 = tot_avg[0] * Wt[1]# + Wb[1]# first class input avg * second class weight + second class bias

    #data input for second class
    wx_10 = tot_avg[1] * Wt[0]# + Wb[0]# second class input avg * first class weight + first class bias
    wx_11 = tot_avg[1] * Wt[1]# + Wb[1]# second class input avg * second class weight + second class bias

    wx_abs = np.zeros(len(wx_00))
    for idx, _ in enumerate(wx_00):
        wx_abs[idx] = np.abs(wx_00[idx] - wx_01[idx]) + np.abs(wx_11[idx] - wx_10[idx])

    selected_idx = np.argsort(wx_abs)[::-1][0:n_selection]
    selected_weights = wx_abs[selected_idx]

    #get evaluation acc from best model
    loss, val_acc = model.evaluate(x_val, y_val)

    K.clear_session()

    return selected_idx, selected_weights, val_acc


def main():
    data = np.loadtxt(r"F:\00_paper_Proj\NeuralNet_GeneSelection\DearWXpub-master\DearWXpub-master\src\dummy.csv" , delimiter = ',')
    xtrain = data[:,:-1]
    ytrain = data[:,-1]
    ytrain = ytrain.astype(np.int)
    ytrain = np.eye(3)[ytrain]
    WxSlp(xtrain,ytrain,xtrain,ytrain,3,WxHyperParameter(learning_ratio=0.001))




if __name__ == "__main__":
    main()

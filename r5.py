from __future__ import print_function
import argparse
import time
import datetime
import json
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
from tflearn.data_utils import image_preloader
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutil import *

import keras

from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Input, Flatten
from keras.layers.core import Dense, Dropout
from keras.regularizers import l2
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, SGD

np.set_printoptions(threshold=sys.maxsize)

def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 200:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
    
def save_history_graph(history):
    
    cdt = datetime.datetime.now()
    finish_time = ('%d-%d-%d_%d:%d'%(cdt.year,cdt.month,cdt.day,cdt.hour,cdt.minute))
    history_logfile = '%s/%s_%s_%s_%s_%s_ts%d_imw%s_epochs%d_%s_%s.json'%(history_dir, dataset_name,exp_name, cnn, classify_type,training_type,train_size,imw,epochs,finish_time,remark)
    
    with open(history_logfile,'w') as f:
        json.dump(str(history.history), f)                

    # Plot training & validation loss values
    image_lossfile = '%s/%s_%s_%s_%s_%s_ts%d_imw%s_epoch%d_%s_%s_loss.png'%(graph_dir, dataset_name,exp_name, cnn, classify_type,training_type,train_size,imw,epochs,finish_time,remark)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss: %s_%s_%s_%s_%s_ts%d_%d'%(dataset_name,exp_name, cnn, classify_type,training_type,train_size,imw))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(image_lossfile)
    plt.clf()
    
    
    if not is_1vs1:
        image_lossfile = '%s/%s_%s_%s_%s_%s_ts%d_imw%s_epoch%d_%s_%s_acc.png'%(graph_dir, dataset_name,exp_name, cnn, classify_type,training_type,train_size,imw,epochs,finish_time,remark)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('acc: %s_%s_%s_%s_%s_ts%d_%d'%(dataset_name,exp_name, cnn, classify_type,training_type,train_size,imw))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(image_lossfile)    
        
    #plt.show()


def train_data():
    start_time = time.time()    
    nClassifiers = num_classes #for 1vsAll

    if is_1vs1:
        mapping, nClassifiers = generate_output_mapping(num_classes)
        trainTargets_ytrain = generate_labels(y_train, mapping)      #Y is an int label, not oneHot
        trainTargets_ytest = generate_labels(y_test, mapping)      #Y is an int label, not oneHot
    
    if not is_train:
        if is_1vs1:
            all_accuracy_last=[]
            all_accuracy_best, model_n = evaluate_1vs1_model(savedmodel_dir,None, x_test,y_test,mapping,trainTargets_ytest,num_classes)    
        else:
            all_accuracy, model_n = evaluate_1vsall_model(savedmodel_dir,None,x_test,y_test)
    
    else:   #train_model
        
        save_dir = os.path.join(os.getcwd(), savedmodel_dir)
        model_name = '%s_%s_ts%s_imw%s_.{epoch:04d}.h5' % (dataset_name,exp_name,train_size,imw)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        
        if data_augmentation:
            print('[INFO==>]  Using real-time data augmentation.')
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False)
            datagen.fit(x_train)
        #end data_augmentation

        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=False)  
        lr_scheduler = LearningRateScheduler(lr_schedule)
        callbacks = [checkpoint,lr_scheduler]     
           
        #create model
        steps_per_epoch=len(x_train)//batch_size if len(x_train)//batch_size > 0 else 1
        
        if is_finetune:
            base_model = ResNet50(weights='imagenet', include_top =False,input_tensor=Input(shape=(imw,imw,3)))
        else:
            base_model = ResNet50(weights=None, include_top =False,input_tensor=Input(shape=(imw,imw,3)))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        #x = Dropout(0.4)(x)
        #x = Dense(int(nodes), activation='relu')(x)   #nodes = 128 or 1024

        #opt=SGD(lr=lr_schedule(0), momentum=0.9)
        opt=Adam(lr=lr_schedule(0))
        optmsg = str(opt)+", start with lr = 1e-3, step 50, metrics=['accuracy']"  

        if is_1vs1:        
            predictions = Dense(nClassifiers, kernel_initializer='he_normal', activation='tanh')(x)
            model = Model(inputs=base_model.input, outputs=predictions)            
            model.compile(loss=custom_crossentropy, optimizer=opt,metrics=[custom_crossentropy])
            checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=False)  
            lr_scheduler = LearningRateScheduler(lr_schedule)
            callbacks = [checkpoint,lr_scheduler]
            model.summary()
            
            history = model.fit_generator(datagen.flow(x_train, trainTargets_ytrain, batch_size=batch_size),
                    validation_data=(x_test, trainTargets_ytest),
                    epochs=epochs, verbose=1, workers=4, shuffle=True, callbacks=callbacks,steps_per_epoch=steps_per_epoch)
            
            all_accuracy_last, model_n = evaluate_1vs1_model(savedmodel_dir,model, x_test,y_test,mapping,trainTargets_ytest,num_classes)
            #all_accuracy_best, model_n = evaluate_1vs1_model(savedmodel_dir,None, x_test,y_test,mapping,trainTargets_ytest,num_classes)
            all_accuracy_best = [0.0, 0.0, 0.0]
            save_history_graph(history)    
                        
        else:   #1vsAll
            predictions = Dense(nClassifiers, kernel_initializer='he_normal', activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs =predictions)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=False)
            lr_scheduler = LearningRateScheduler(lr_schedule)
            callbacks = [checkpoint,lr_scheduler]
            model.summary()
            
            history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs, verbose=1, workers=4, shuffle=True, callbacks=callbacks,steps_per_epoch=steps_per_epoch)

            all_accuracy, model_n = evaluate_1vsall_model(savedmodel_dir,model,x_test,y_test)
            save_history_graph(history)

        print('INFO==> : ', optmsg)            
                
    #ending if is_train, so either train or just test model, writeTOLogfile
    end_time = time.time()
    total_time = (int)(end_time-start_time)/60.0
    if(is_1vs1):
        write_to_logfile_1vs1(logfile_report, total_time, dataset_name,exp_name,train_size,imw,epochs,batch_size,remark, model_n, all_accuracy_last, all_accuracy_best)
    else:
        write_to_logfile_1vsa(logfile_report, total_time, dataset_name,exp_name,train_size,imw,epochs,batch_size,remark, model_n, all_accuracy)

            

def print_working_detail():
    print('_____________working detail____________')
    print('%s, %s ,%s      '%(cnn, training_type,classify_type))
    print("dataset:     ", dataset_name)
    print("train_size:  ", train_size)
    print("exp:         ", exp_name)
    print("epochs:      ", epochs)
    print("batch_size:  ", batch_size)
    print("imw:         ", imw)
    print('logfile_report:', logfile_report)
    print('savedmodel_dir:', savedmodel_dir)    
    print("weight        : ", weight)
    print('_______________________________________')    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TrainingModel")
    parser.add_argument('--dataset_name',   default ='dataset_demo5', help='dataset name')
    parser.add_argument('--exp_name',       default ='exp1', help='exp name')
    parser.add_argument('--cnn',            default = 'resnet50', help='models')
    parser.add_argument('--is_finetune',    type=str2bool, default=False)    
    parser.add_argument('--epochs',         default=10, type=int)
    parser.add_argument('--batch_size',     default=16, type = int)
    parser.add_argument('--train_size', 	default=100, type=int)
    parser.add_argument('--imw',            default=299, type=int, help='Image size resnet(224), inception(299)')
    parser.add_argument('--is_train',       type=str2bool, default=True)
    parser.add_argument('--is_1vs1',        type=str2bool, default=True)
    parser.add_argument('--workat',         default='p')
    parser.add_argument('--remark',         default='r5', help='remark for savemodel folder')
    
    args = parser.parse_args()
    
    #assign arguments to global variables

    dataset_name    = args.dataset_name
    exp_name        = args.exp_name
    cnn             = args.cnn
    is_finetune     = args.is_finetune
    imw             = args.imw
    epochs          = args.epochs
    batch_size      = args.batch_size
    train_size		= args.train_size
    is_train        = args.is_train
    is_1vs1         = args.is_1vs1
    workat          = args.workat
    remark          = args.remark

    subtract_pixel_mean = True
    data_augmentation = True
    num_channels    = 3

    
    d3 = dataset_name[-3]
    d2 = dataset_name[-2]
    num_classes = int(dataset_name[-3:]) if d3.isdigit() else int(dataset_name[-2:]) if d2.isdigit() else int(dataset_name[-1:])
    
    if is_1vs1:
        classify_type = '1vs1'
    else:
        classify_type = '1vsa'
        
    if is_finetune:
        training_type = 'finetune'
        weight = 'imagenet'
    else:
        training_type = 'scratch'   
        weight = None

    
    logfile_dir ='_allresult_%s_%s_%s'%(classify_type,training_type,cnn)
    if not os.path.isdir(logfile_dir):
        os.makedirs(logfile_dir)
    
    if workat=='office':
        logfile_report = '%s/%s_%s_%s_%s_%s.log'%(logfile_dir,dataset_name,classify_type,cnn,training_type,remark)
        savedmodel_dir = '/project/ppawara/backup_1vs1_revision/savedmodel/savedmodel_%s_%s_ts%d_%s_%s_%s_%d_%s'%(dataset_name,classify_type,train_size,exp_name,cnn,training_type,imw,remark)
        history_dir    = 'history_log'
        graph_dir      = 'graph_%s'%(dataset_name)
        
    elif workat=='colab':
        scratch_dir    = '/content/drive/My Drive/'
        logfile_report = '%s%s/%s_%s_%s_%s_%s.log'%(scratch_dir,logfile_dir,dataset_name,classify_type,cnn,training_type,remark)
        savedmodel_dir = '%ssavedmodel/savedmodel_%s_%s_ts%d_%s_%s_%s_%d_%s'%(scratch_dir,dataset_name,classify_type,train_size,exp_name,cnn,training_type,imw,remark)
        history_dir    = '%shistory_log'%(scratch_dir)
        graph_dir      = '%sgraph/graph_%s'%(scratch_dir,dataset_name)
        
    else: #peregrine
        scratch_dir = '/scratch/p274641/1vs1_savebest/'        
        logfile_report = '%s/%s_%s_%s_%s_%s.log'%(logfile_dir,dataset_name,classify_type,cnn,training_type,remark)
        savedmodel_dir = '%ssavedmodel_%s_%s_ts%d_%s_%s_%s_%d_%s'%(scratch_dir,dataset_name,classify_type,train_size,exp_name,cnn,training_type,imw,remark)
        history_dir    = 'history_log'
        graph_dir      = 'graph_%s'%(dataset_name)    

    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)

    print('logfile_report : ', logfile_report)        
    print('savedmodel_dir : ', savedmodel_dir)
    print('history_dir    : ', history_dir)
    print('graph_dir      : ', graph_dir) 
    
    print_working_detail()

    x_train,y_train, x_test, y_test  = load_dataset(dataset_name, exp_name, imw, num_classes,train_size,is_1vs1,workat)
   
    
    train_data()

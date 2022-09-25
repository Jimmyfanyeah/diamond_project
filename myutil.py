from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import time
import argparse
import sys
from keras.models import load_model
import time
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from keras import backend as K
from tflearn.data_utils import image_preloader
from scipy import stats
import csv


#np.set_printoptions(threshold=np.nan)
def str2bool(v):
    return v.lower() in ("yes","true")


def write_to_logfile(logfile_report, total_time, dataset_name,exp_name,train_size,imw,epochs,batch_size, accuracy):
    
    #save to log file
    with open(logfile_report,'a') as logfile:

        logfile.write('%s, %s, ts:%s, imw:%s, epochs:%s, bs: %s, time: %.2f, acc: %.3f' %(dataset_name, exp_name, train_size, imw,epochs,batch_size, total_time, accuracy))

        logfile.write("\n")
        if(exp_name=='exp5'):
            logfile.write("\n") 

    #save to csv file #
    logfile_dir     = logfile_report.split('/',1)[0]+'/'      #to get logfile_dir (_allresults_1vsa_scratch_inception)
    csvfile_report  = logfile_dir+dataset_name[8:]+'.csv'        #make it short filename --> short tab name in excel file
    with open(csvfile_report,'a') as csvfile:
        fieldnames =['dataset_name','exp_name','train_size','imw','epochs','batch_size','total_time','accuracy']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        #writer.writeheader()
        writer.writerow({'dataset_name':dataset_name, 'exp_name':exp_name,'train_size':train_size,'imw':imw,'epochs':epochs,'batch_size':batch_size,'total_time':total_time, 'accuracy':accuracy})
        if(exp_name=='exp5'):
            writer.writerow({}) 


def ovo_crossentropy(y_true, y_pred):
    y_true = (y_true+1.0)/2.0
    y_pred = (y_pred+1.0)/2.0

    y_pred = tf.where(y_pred > 0.00001, y_pred, tf.zeros_like(y_pred)+0.00001)
    y_pred = tf.where(y_pred < 0.99999, y_pred, tf.zeros_like(y_pred)+0.99999)

    ovo_loss = tf.reduce_mean((-(y_true*tf.log(y_pred)) - ((1-y_true)*tf.log((1-y_pred)))), axis=1)
    return ovo_loss
    
    
def evaluate_1vsall_model(savedmodel_dir, model, x_test, y_test):

    #if model is Null (test only), then load the saved model
    if model==None:
        cwd = os.getcwd()
        os.chdir(savedmodel_dir)
        sortmodel = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
        for i,model_name in enumerate(reversed(sortmodel)):
            print ('[INFO] loading 1vsa model: %s'% (model_name))
            model = load_model(str(model_name))
            os.chdir(cwd)
            break
    
    print("[INFO] y_test class")
    print(y_test) 

    # Predict from last model
    scores = model.evaluate(x_test, y_test, verbose=1)
    y_pred =model.predict(x_test)
    print("[INFO] y_pred...")
    print(np.argmax(y_pred,1))

    loss = scores[0]
    accuracy  = scores[1]*100
    print('\n[INFO] accuracy : %f'% (accuracy))


    return accuracy


def evaluate_1vs1_model(savedmodel_dir, model, x_test,y_test,mapping,trainTargets_ytest,num_classes):

    #if model is Null (test only), then load the saved model
    if model==None: 
        cwd = os.getcwd()
        os.chdir(savedmodel_dir)
        sortmodel = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
        for i,model_name in enumerate(reversed(sortmodel)):
            print ('[INFO] loading 1vs1 model...: %s'% (model_name))
            model = load_model(str(model_name),custom_objects={"ovo_crossentropy": ovo_crossentropy})
            os.chdir(cwd)
            break
    

    y_output=model.predict(x_test)          #got y_output as 45 index
    y_pred  = np.matmul(y_output,mapping)    #mapping to one-hot (10 class)
    

    #print(np.argmax(y_pred,1))              #cast to class number
    correct_pred = np.equal(np.argmax(y_pred,1), y_test)
    accuracy = ((correct_pred.astype(float)).mean())*100

    #print('[INFO] ==> y_output')  
    #print(y_output)


    print("\n[INFO] y_test class")
    print(y_test)  

    print("\n[INFO] y_pred class")
    print(np.argmax(y_pred,1))              #cast to class number
    
    print("Accuracy: %.3f "% (accuracy))


    return accuracy

    

def generate_output_mapping(num_classes):
  
  nClassifiers = int(num_classes * (num_classes - 1) / 2)
  mapping = np.zeros((nClassifiers, num_classes), dtype = "float")
  #print("shape----",mapping.shape)
  classifierCounter = 0
  
  for labelA in range(num_classes):
    for labelB in range(labelA + 1, num_classes):
      if labelA + labelB == 0:
        continue
      mapping[classifierCounter][labelA] = 1
      mapping[classifierCounter][labelB] = -1
      classifierCounter += 1
  return mapping, nClassifiers

def get_train_label(mapping, label):

  return np.transpose(mapping)[label]

def generate_labels(labels, mapping):
  targets = np.zeros((len(labels), len(mapping)))
  for labelIdx in range(len(labels)):
    targets[labelIdx] = get_train_label(mapping, labels[labelIdx])
  return targets

def random_shuffle(images, labels):
    perm = np.arange(len(images))
    np.random.shuffle(perm)
    random_image = images[perm]
    random_label = labels[perm]
    return random_image, random_label
        
def load_dataset(dataset_name, exp_name, imw,num_classes, train_size, is_1vs1):
    
    np.set_printoptions(threshold=sys.maxsize)
    
    work_dir='/data/work_2018_tf/dataset_ican/'+dataset_name+"/"+exp_name+"/"

    train_dir = work_dir+'train'
    test_dir = work_dir+'test'
    
    if is_1vs1:         
        x_train, y_train = image_preloader(train_dir, image_shape=(imw, imw),  grayscale=False,  mode='folder', categorical_labels=False, normalize=True)
        x_test, y_test = image_preloader(test_dir, image_shape=(imw, imw), grayscale=False, mode='folder', categorical_labels=False,  normalize=True)
        
    else:
        x_train, y_train = image_preloader(train_dir, image_shape=(imw, imw),  grayscale=False,  mode='folder', categorical_labels=True, normalize=True)
        x_test, y_test = image_preloader(test_dir, image_shape=(imw, imw), grayscale=False, mode='folder', categorical_labels=True,  normalize=True)  
           
    x_train = np.asarray(x_train[:])
    y_train = np.asarray(y_train[:])
    x_test = np.asarray(x_test[:])
    y_test = np.asarray(y_test[:])
    
    original_training=len(y_train)
    print("The whole training set contain %i image"%(original_training))
    #print(y_test)
    
    #shuffle data and split traindata to train_size subset
    if train_size != 100:
        x_train,y_train = random_shuffle(x_train,y_train)
        num_train = len(x_train)
        train_subset = num_train*train_size //100
        x_train = x_train[:train_subset]
        y_train = y_train[:train_subset]
              
        
        #print("__Training data after subsplit___")
        #print(y_train)
        print("10 samples of training label")
        for i in range(10):
            print(y_train[i])
    
            
    print("working on {0}% of training set(whole {1} images), with {2} training images.".format(train_size, original_training, len(y_train)))
    print("Test on {0} test images".format(len(x_test)))
    
    
    #subtract xtrain mean
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    
    return x_train,y_train, x_test, y_test

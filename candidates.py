from utils import gather_2_5D_data
from gather_patches import *
import tensorflow as tf
import numpy as np
import os, argparse
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
DATA_DIR = '/home/user/tony/DATA/np/ct'

def candidate_model_fn(features, labels, mode):
    w = 25
    input_layer = tf.reshape(features["patch"], [-1, w, w, 1])
    xyz = tf.reshape(features["xyz"], [-1, 3, 1], name='location_features')
    
    num_conv_layers = (w-1)//2
    
    for i in range(1,num_conv_layers):
        conv = tf.layers.conv2d(inputs=input_layer,
                                filters=16,
                                kernel_size=[3,3],
                                padding='valid',
                                activation=tf.nn.relu,
                                name=f'conv_{i}')
    conv = tf.layers.conv2d(inputs=conv,
                                filters=32,
                                kernel_size=[3,3],
                                padding='valid',
                                activation=tf.nn.relu,
                                name=f'texture_features')
    
    fc = tf.concat([xyz, conv], axis=0, name='feature_vector')
    fc = tf.nn.dropout(fc, 0.5)
    hidden = tf.layers.dense(fc, units=192, activation=tf.nn.relu, name='hidden')
    hidden = tf.nn.dropout(hidden, 0.5)
    
    logits = tf.layers.dense(hidden, units=2, activation=None, name='logits')
    predictions = {'classes': tf.argmax(input=logits, axis=1),
                   'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    lr = tf.train.exponential_decay(1e-3, tf.train.get_global_step(), 2000, 0.2, staircase=True)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metrics_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metrics_ops=eval_metrics_ops)


def train(num_patients):
    if not num_patients:
        num_patients = 1
    images, locs, labels = collect_patches(num_patients)
    
    config = tf.estimator.RunConfig(log_step_count_steps=100)
    classifier = tf.estimator.Estimator(model_fn=candidate_model_fn,
                                              config=config,
                                              model_dir="./model")
    
    def train_input_fn(images, locs, labels):
        return tf.estimator.inputs.numpy_input_fn(x={"patch":images,
                                                     "xyz":locs},
                                                  y=labels,
                                                  batch_size=128,
                                                  num_epochs=10,
                                                  shuffle=True)
    
    print("TRAINING STARTED")
    classifier.train(input_fn=train_input_fn(images, locs,labels))
    print("FINISHED TRAINING")


def test(num_patients):
    if not num_patients:
        num_patients = 1
    images, locs, labels = collect_patches(num_patients)
    
    config = tf.estimator.RunConfig(log_step_count_steps=100)
    classifier = tf.estimator.Estimator(model_fn=candidate_model_fn,
                                              config=config,
                                              model_dir="./model")
    
    def eval_input_fn(images, locs, labels):
        return tf.estimator.inputs.numpy_input_fn(x={"patch":images,
                                                     "xyz":locs},
                                                  y=labels,
                                                  num_epochs=1,
                                                  shuffle=False)
    results = classifier.evaluate(input_fn=eval_input_fn(images, locs, labels))
    print(f'Results: {results}')
    

def main(unused_argv):
    parser = argparse.ArgumentParser(description="Train or Test 2.5D CNN for CAC Candidate Classification")
    parser.add_argument("-tr", "--train", type=int, default=0, help="train model with n patient's data")
    parser.add_argument("-te", "--test", type=int, default=0, help="test model with n patient's data")
    args = parser.parse_args()
    print("Started candidates.py")
    if args.train:
        print(f'Will train model on {args.train} patients')
        train(args.train)
    if args.test:
        print(f'Will test model on {args.test} patients')
        test(args.train)
        
        
if __name__ == '__main__':
    tf.app.run()
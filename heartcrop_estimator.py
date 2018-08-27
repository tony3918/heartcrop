from AlexNet import AlexNet
from utils import gather_2_5D_data
import tensorflow as tf
import numpy as np
import os, argparse
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
DATA_DIR = '/home/user/tony/DATA/np/ct'

def heartcrop_model_fnn(features, labels, mode):
    input = tf.reshape(features["x"], [-1,256,256,1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        model = AlexNet(is_training=True, log=False)
    else:
        model = AlexNet(is_training=False, log=False)
    prediction = model.inference(input)

    predictions = {"classes":tf.argmax(input=prediction, axis=1)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=prediction)

    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = tf.train.exponential_decay(1e-3, tf.train.get_global_step(), 2000,
                                        0.2, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train(num_patients):
    if not num_patients:
        num_patients = 50
    patient_list = sorted(os.listdir(DATA_DIR))[:num_patients]
    (train_images_axial, train_labels_axial,
     train_images_sagital, train_labels_sagital,
     train_images_coronal, train_labels_coronal) = gather_2_5D_data(patient_list, training=False)

    print(f'Train image shape: {train_images_coronal.shape}')
    print(f'Train image type: {train_images_coronal.dtype}')
    print(f'Train label shape: {train_labels_coronal.shape}')
    print(f'Train label type: {train_labels_coronal.dtype}')


    config = tf.estimator.RunConfig(log_step_count_steps=100)
    classifier_axial = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                              config=config,
                                              model_dir="./models/axial")
    classifier_sagital = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                                config=config,
                                                model_dir="./models/sagital")
    classifier_coronal = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                                config=config,
                                                model_dir="./models/coronal")

    def train_input_fn(images, labels):
        return tf.estimator.inputs.numpy_input_fn(x={"x":images},
                                                  y=labels,
                                                  batch_size=128,
                                                  num_epochs=10,
                                                  shuffle=True)
    print("TRAINING AXIAL")
    classifier_axial.train(input_fn=train_input_fn(train_images_axial, train_labels_axial))
    print("TRAINING SAGITAL")
    classifier_sagital.train(input_fn=train_input_fn(train_images_sagital, train_labels_sagital))
    print("TRAINING CORONAL")
    classifier_coronal.train(input_fn=train_input_fn(train_images_coronal, train_labels_coronal))
    print("FINISHED TRAINING")


def test(num_patients):
    if not num_patients:
        num_patients = 12
    patient_list = sorted(os.listdir(DATA_DIR))[-num_patients:]
    (test_images_axial, test_labels_axial,
     test_images_sagital, test_labels_sagital,
     test_images_coronal, test_labels_coronal) = gather_2_5D_data(patient_list)

    print(f'Test image shape: {test_images_coronal.shape}')
    print(f'Test image type: {test_images_coronal.dtype}')
    print(f'Test label shape: {test_labels_coronal.shape}')
    print(f'Test label type: {test_labels_coronal.dtype}')

    config = tf.estimator.RunConfig(log_step_count_steps=100)
    classifier_axial = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                              config=config,
                                              model_dir="./models/axial")
    classifier_sagital = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                                config=config,
                                                model_dir="./models/sagital")
    classifier_coronal = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                                config=config,
                                                model_dir="./models/coronal")

    def eval_input_fn(images, labels):
        return tf.estimator.inputs.numpy_input_fn(x={"x":images},
                                                  y=labels,
                                                  num_epochs=1,
                                                  shuffle=False)

    results_axial = classifier_axial.evaluate(input_fn=eval_input_fn(test_images_axial, test_labels_axial))
    results_sagital = classifier_sagital.evaluate(input_fn=eval_input_fn(test_images_sagital, test_labels_sagital))
    results_coronal = classifier_coronal.evaluate(input_fn=eval_input_fn(test_images_coronal, test_labels_coronal))
    print(f'Axial: {results_axial}')
    print(f'Sagital: {results_sagital}')
    print(f'Coronal: {results_coronal}')


def main(unused_argv):
    parser = argparse.ArgumentParser(description="Train or Test 2.5D CNN HeartCrop")
    parser.add_argument("-tr", "--train", type=int, default=0, help="train model with n patient's data")
    parser.add_argument("-te", "--test", type=int, default=0, help="test model with n patient's data")
    args = parser.parse_args()
    print("Started heartcrop_estimator.py")
    if args.train:
        print(f'Will train model on {args.train} patients')
        train(args.train)
    if args.test:
        print(f'Will test model on {args.test} patients')
        test(args.train)


if __name__ == '__main__':
    tf.app.run()

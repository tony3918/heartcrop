from AlexNet import AlexNet
from utils import gather_2_5D_data
import tensorflow as tf
import numpy as np
import os, time

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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


def train():
    DATA_DIR = '/home/user/tony/DATA/np/ct'
    images_axial = []
    images_sagital = []
    images_coronal = []
    for patient_id in sorted(os.listdir(DATA_DIR))[:50]:
        ct = np.load(os.path.join(DATA_DIR, patient_id))
        images_axial.append(ct)
        images_sagital.append(np.transpose(ct, [2, 0, 1]))
        images_coronal.append(np.transpose(ct, [1, 0, 2]))
    images_axial = np.concatenate([ct for ct in images_axial]).astype(np.float32)
    images_sagital = np.concatenate([ct for ct in images_sagital]).astype(np.float32)
    images_coronal = np.concatenate([ct for ct in images_coronal]).astype(np.float32)

    labels_axial = []
    labels_sagital = []
    labels_coronal = []
    with open('/home/user/tony/CODE/heartcrop_estimator/true_roi.txt', 'r') as f:
        for line in f.readlines()[:50]:
            id, x1, y1, z1, x2, y2, z2 = [int(i) for i in line.strip().split('\t')]
            label = np.zeros([256, 256, 256], dtype=np.int8)
            label[z1:z2, y1:y2, x1:x2] = 1
            labels_axial.append(label)
            labels_sagital.append(np.transpose(label, [2, 0, 1]))
            labels_coronal.append(np.transpose(label, [1, 0, 2]))
    labels_axial = np.any(np.concatenate([label for label in labels_axial]), axis=(1, 2)).astype(np.int32)
    labels_sagital = np.any(np.concatenate([label for label in labels_sagital]), axis=(1, 2)).astype(np.int32)
    labels_coronal = np.any(np.concatenate([label for label in labels_coronal]), axis=(1, 2)).astype(np.int32)

    print(f'Train image shape: {images_axial.shape}')
    print(f'Train image type: {images_axial.dtype}')
    print(f'Train label shape: {labels_axial.shape}')
    print(f'Train label type: {labels_axial.dtype}')

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
                                                  num_epochs=30,
                                                  shuffle=True)
    print("TRAINING AXIAL")
    classifier_axial.train(input_fn=train_input_fn(images_axial, labels_axial))
    print("TRAINING SAGITAL")
    classifier_sagital.train(input_fn=train_input_fn(images_sagital, labels_sagital))
    print("TRAINING CORONAL")
    classifier_coronal.train(input_fn=train_input_fn(images_coronal, labels_coronal))
    print("FINISHED TRAINING")


def test():
    DATA_DIR = '/home/user/tony/DATA/np/ct'
    images_axial = []
    images_sagital = []
    images_coronal = []
    for patient_id in sorted(os.listdir(DATA_DIR))[50:]:
        ct = np.load(os.path.join(DATA_DIR, patient_id))
        images_axial.append(ct)
        images_sagital.append(np.transpose(ct, [2, 0, 1]))
        images_coronal.append(np.transpose(ct, [1, 0, 2]))
    images_axial = np.concatenate([ct for ct in images_axial]).astype(np.float32)
    images_sagital = np.concatenate([ct for ct in images_sagital]).astype(np.float32)
    images_coronal = np.concatenate([ct for ct in images_coronal]).astype(np.float32)

    labels_axial = []
    labels_sagital = []
    labels_coronal = []
    with open('/home/user/tony/CODE/heartcrop_estimator/true_roi.txt', 'r') as f:
        for line in f.readlines()[50:]:
            id, x1, y1, z1, x2, y2, z2 = [int(i) for i in line.strip().split('\t')]
            label = np.zeros([256, 256, 256], dtype=np.int8)
            label[z1:z2, y1:y2, x1:x2] = 1
            labels_axial.append(label)
            labels_sagital.append(np.transpose(label, [2, 0, 1]))
            labels_coronal.append(np.transpose(label, [1, 0, 2]))
    labels_axial = np.any(np.concatenate([label for label in labels_axial]), axis=(1, 2)).astype(np.int32)
    labels_sagital = np.any(np.concatenate([label for label in labels_sagital]), axis=(1, 2)).astype(np.int32)
    labels_coronal = np.any(np.concatenate([label for label in labels_coronal]), axis=(1, 2)).astype(np.int32)

    print(f'Test image shape: {images_axial.shape}')
    print(f'Test image type: {images_axial.dtype}')
    print(f'Test label shape: {labels_axial.shape}')
    print(f'Test label type: {labels_axial.dtype}')

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

    results_axial = classifier_axial.evaluate(input_fn=eval_input_fn(images_axial, labels_axial))
    results_sagital = classifier_axial.evaluate(input_fn=eval_input_fn(images_sagital, labels_sagital))
    results_coronal = classifier_axial.evaluate(input_fn=eval_input_fn(images_coronal, labels_coronal))
    print(f'Axial: {results_axial}')
    print(f'Sagital: {results_sagital}')
    print(f'Coronal: {results_coronal}')



def main(unused_argv):
    train()
    test()


if __name__ == '__main__':
    tf.app.run()
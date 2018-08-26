from AlexNet import AlexNet
from utils import gather_2_5D_data
import tensorflow as tf
import numpy as np
import os, time

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def heartcrop_model_fnn(features, labels, mode):
    input_axial = tf.reshape(features["x"], [-1,256,256,1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        model_axial = AlexNet(is_training=True, name='axial', log=False)
    else:
        model_axial = AlexNet(is_training=False, name='axial', log=False)
    prediction_axial = model_axial.inference(input_axial)

    predictions = {"classes":tf.argmax(input=prediction_axial, axis=1)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=prediction_axial)

    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = tf.train.exponential_decay(1e-3, tf.train.get_global_step(), 2000,
                                        0.2, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train():
    DATA_DIR = '/home/user/tony/DATA/np/ct'
    images_axial = []
    for patient_id in sorted(os.listdir(DATA_DIR))[:50]:
        ct = np.load(os.path.join(DATA_DIR, patient_id))
        images_axial.append(ct)
    images_axial = np.concatenate([ct for ct in images_axial]).astype(np.float32)

    labels_axial = []
    with open('/home/user/tony/CODE/heartcrop_estimator/true_roi.txt', 'r') as f:
        for line in f.readlines()[:50]:
            id, x1, y1, z1, x2, y2, z2 = [int(i) for i in line.strip().split('\t')]
            label = np.zeros([256, 256, 256], dtype=np.int8)
            label[z1:z2, y1:y2, x1:x2] = 1
            labels_axial.append(label)
    labels_axial = np.any(np.concatenate([label for label in labels_axial]), axis=(1, 2)).astype(np.int32)

    print(f'Train image shape: {images_axial.shape}')
    print(f'Train image type: {images_axial.dtype}')
    print(f'Train label shape: {labels_axial.shape}')
    print(f'Train label type: {labels_axial.dtype}')

    config = tf.estimator.RunConfig(log_step_count_steps=100)
    classifier = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                        config=config,
                                        model_dir="./model")
    # tensors_to_log = {}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
    #                                           every_n_iter=200)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":images_axial},
                                                        y=labels_axial,
                                                        batch_size=128,
                                                        num_epochs=10,
                                                        shuffle=True)
    print("READY FOR TRAINING")
    classifier.train(input_fn=train_input_fn)
    print("FINISHED TRAINING")
    print("******************************************************")
    print("******************************************************")


def test():
    DATA_DIR = '/home/user/tony/DATA/np/ct'
    images_axial = []
    for patient_id in sorted(os.listdir(DATA_DIR))[50:]:
        ct = np.load(os.path.join(DATA_DIR, patient_id))
        images_axial.append(ct)
    images_axial = np.concatenate([ct for ct in images_axial]).astype(np.float32)

    labels_axial = []
    with open('/home/user/tony/CODE/heartcrop_estimator/true_roi.txt', 'r') as f:
        for line in f.readlines()[50:]:
            id, x1, y1, z1, x2, y2, z2 = [int(i) for i in line.strip().split('\t')]
            label = np.zeros([256, 256, 256], dtype=np.int8)
            label[z1:z2, y1:y2, x1:x2] = 1
            labels_axial.append(label)
    labels_axial = np.any(np.concatenate([label for label in labels_axial]), axis=(1, 2)).astype(np.int32)

    print(f'Test image shape: {images_axial.shape}')
    print(f'Test image type: {images_axial.dtype}')
    print(f'Test label shape: {labels_axial.shape}')
    print(f'Test label type: {labels_axial.dtype}')

    config = tf.estimator.RunConfig(log_step_count_steps=100)
    classifier = tf.estimator.Estimator(model_fn=heartcrop_model_fnn,
                                        config=config,
                                        model_dir="./model")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":images_axial},
                                                       y=labels_axial,
                                                       num_epochs=1,
                                                       shuffle=False)
    results = classifier.evaluate(input_fn=eval_input_fn)
    print(results)



def main(unused_argv):
    # train()
    test()


if __name__ == '__main__':
    tf.app.run()
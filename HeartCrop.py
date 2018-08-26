from AlexNet import AlexNet
from utils import gather_2_5D_data
import tensorflow as tf
import numpy as np
import os, time
import matplotlib.pyplot as plt

def HeartCrop(BATCH_SIZE=None, patient_id=None):
    RAW_DATA_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\CV_RISK\DATA\raw'
    NP_DATA_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\CV_RISK\DATA\np'
    CKPT_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\HEART_CROP\checkpoints'
    SUMMARY_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\HEART_CROP\checkpoints\summary'
    
    MODEL_NAME = 'heartcropcnn-20'
    THRESHOLD = 0.5
    
    if not patient_id:
        patient_id = '4386989'
    (images_axial, images_sagital, images_coronal) = gather_2_5D_data([patient_id], training=False)
    print(f'Starting HeartCrop for patient {patient_id}')
    
    if not BATCH_SIZE:
        BATCH_SIZE = 16
    NUM_SAMPLES = len(images_axial)
    NUM_BATCHES = NUM_SAMPLES // BATCH_SIZE
    
    
    input_axial = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 1])
    input_sagital = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 1])
    input_coronal = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 1])
    
    model_axial = AlexNet(is_training=False, name='axial')
    model_sagital = AlexNet(is_training=False, name='sagital')
    model_coronal = AlexNet(is_training=False, name='coronal')
    
    prediction_axial = model_axial.inference(input_axial)
    prediction_sagital = model_sagital.inference(input_sagital)
    prediction_coronal = model_coronal.inference(input_coronal)
    
    saver = tf.train.Saver()
    
    prediction_ax = np.array([])
    prediction_sag = np.array([])
    prediction_cor = np.array([])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(f'Restoring HeartCrop model <{MODEL_NAME}>')
        saver.restore(sess, os.path.join(CKPT_DIR, MODEL_NAME))
        
        start = time.time()
        for batch_idx in range(NUM_BATCHES):
            batch_images_axial = images_axial[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE,:,:]
            batch_images_axial = np.reshape(batch_images_axial, [BATCH_SIZE, 256, 256, 1])
            
            batch_images_sagital = images_sagital[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE,:,:]
            batch_images_sagital = np.reshape(batch_images_sagital, [BATCH_SIZE, 256, 256, 1])
            
            batch_images_coronal = images_coronal[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE,:,:]
            batch_images_coronal = np.reshape(batch_images_coronal, [BATCH_SIZE, 256, 256, 1])

            feed = {input_axial:batch_images_axial,
                        input_sagital:batch_images_sagital,
                        input_coronal:batch_images_coronal}

            pred_ax, pred_sag, pred_cor = sess.run([prediction_axial, prediction_sagital, prediction_coronal], feed) 
           
            pred_ax = tf.nn.softmax(pred_ax).eval()
            pred_sag = tf.nn.softmax(pred_sag).eval()
            pred_cor = tf.nn.softmax(pred_cor).eval()
            
            prediction_ax = np.append(prediction_ax, pred_ax[:,1])
            prediction_sag = np.append(prediction_sag, pred_sag[:,1])
            prediction_cor = np.append(prediction_cor, pred_cor[:,1])
            
#            print(pred_ax)
    roi_pos_ax = find_roi_pos(prediction_ax, THRESHOLD)
    roi_pos_sag = find_roi_pos(prediction_sag, THRESHOLD)
    roi_pos_cor = find_roi_pos(prediction_cor, THRESHOLD)
    print(f'HEART ROI POSITION FOR PATIENT {patient_id}')
    print(f'x: {roi_pos_sag}')
    print(f'y: {roi_pos_cor}')
    print(f'z: {roi_pos_ax}')
#    heart_roi = np.zeros([256, 256, 256], dtype=np.bool_)
#    heart_roi[roi_pos_ax[0]:roi_pos_ax[1],
#              roi_pos_cor[0]:roi_pos_cor[1],
#              roi_pos_sag[0]:roi_pos_sag[1]] = True
    
#    for z in range(0,256,16):
#        images_axial[heart_roi==1] = 0
##        plt.title(f'roi slice {z}')
##        plt.imshow(heart_roi[z,:,:], cmap='gray')
##        plt.show()
#        plt.title(f'ct slice {z}')
#        plt.imsave(os.path.join(r'C:\Users\USER\Desktop\test',f'{z}.png'),images_axial[z,:,:], cmap='gray')
##        plt.show()
    print('Ending HeartCrop')
    print()
    return roi_pos_ax, roi_pos_cor, roi_pos_sag
    
def find_roi_pos(prediction, THRESHOLD):
    prediction = (prediction > THRESHOLD).astype(np.int8)
    start_pos = 0
    roi_start_pos = 0
    roi_end_pos = 0
    for pos in range(1,len(prediction)):
        if prediction[pos] - prediction[pos-1] == 1:
            start_pos = pos
        if prediction[pos] - prediction[pos-1] == -1:
            if pos - start_pos > roi_end_pos - roi_start_pos:
                roi_start_pos = start_pos
                roi_end_pos = pos
    return roi_start_pos, roi_end_pos
            
    
if __name__ == '__main__':
    HeartCrop()
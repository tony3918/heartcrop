import numpy as np
import os

#RAW_DATA_DIR = '/home/user/tony/DATA/_original'
#NP_DATA_DIR = '/home/user/tony/DATA/np'
#CKPT_DIR = '/home/user/tony/CODE/HEART_CROP/checkpoints'
#SUMMARY_DIR = '/home/user/tony/CODE/HEART_CROP/checkpoints/summary'

RAW_DATA_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\CV_RISK\DATA\raw'
NP_DATA_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\CV_RISK\DATA\np'
CKPT_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\HEART_CROP\checkpoints'
SUMMARY_DIR = r'C:\Users\USER\Documents\연세대학교\Research\Heart_CT\HEART_CROP\checkpoints\summary'

def gather_2_5D_data(patient_id_list, training=True):
    images_axial = []
    images_sagital = []
    images_coronal = []
    for patient_id in patient_id_list:
        ct = np.load(os.path.join(NP_DATA_DIR,'ct',patient_id)+'.npy')
        images_axial.append(ct)
        images_sagital.append(np.transpose(ct,[2,0,1]))
        images_coronal.append(np.transpose(ct,[1,0,2]))
    select = np.arange(0,256*len(patient_id_list))
    images_axial=np.concatenate([ct for ct in images_axial])[select]
    images_sagital=np.concatenate([ct for ct in images_sagital])[select]
    images_coronal=np.concatenate([ct for ct in images_coronal])[select]
    
    if training:
        labels_axial = []
        labels_sagital = []
        labels_coronal = []
        for patient_id in patient_id_list:
            label = np.load(os.path.join(NP_DATA_DIR,'label',patient_id)+'.npy')
            labels_axial.append(label)
            labels_sagital.append(np.transpose(label,[2,0,1]))
            labels_coronal.append(np.transpose(label,[1,0,2]))
        select = np.arange(0,256*len(patient_id_list))
        labels_axial = np.eye(2)[np.any(np.concatenate([label for label in labels_axial]), axis=(1,2)).astype(np.int8)][select]
        labels_sagital = np.eye(2)[np.any(np.concatenate([label for label in labels_sagital]), axis=(1,2)).astype(np.int8)][select]
        labels_coronal = np.eye(2)[np.any(np.concatenate([label for label in labels_coronal]), axis=(1,2)).astype(np.int8)][select]
        return images_axial, labels_axial, images_sagital, labels_sagital, images_coronal, labels_coronal
    else:
        return images_axial, images_sagital, images_coronal
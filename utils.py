import numpy as np
import os

DATA_DIR = '/home/user/tony/DATA/np/ct'

def gather_2_5D_data(patient_id_list, training=True):
    images_axial = []
    images_sagital = []
    images_coronal = []
    for patient_id in patient_id_list:
        ct = np.load(os.path.join(DATA_DIR,patient_id))
        images_axial.append(ct)
        images_sagital.append(np.transpose(ct,[2,0,1]))
        images_coronal.append(np.transpose(ct,[1,0,2]))
    images_axial = np.concatenate([ct for ct in images_axial]).astype(np.float32)
    images_sagital = np.concatenate([ct for ct in images_sagital]).astype(np.float32)
    images_coronal = np.concatenate([ct for ct in images_coronal]).astype(np.float32)
    
    if training:
        labels_axial = []
        labels_sagital = []
        labels_coronal = []
        for patient_id in patient_id_list:
            label = np.load(os.path.join(DATA_DIR,patient_id))
            labels_axial.append(label)
            labels_sagital.append(np.transpose(label,[2,0,1]))
            labels_coronal.append(np.transpose(label,[1,0,2]))
        labels_axial = np.any(np.concatenate([label for label in labels_axial]), axis=(1, 2)).astype(np.int32)
        labels_sagital = np.any(np.concatenate([label for label in labels_sagital]), axis=(1, 2)).astype(np.int32)
        labels_coronal = np.any(np.concatenate([label for label in labels_coronal]), axis=(1, 2)).astype(np.int32)
        return images_axial, labels_axial, images_sagital, labels_sagital, images_coronal, labels_coronal
    else:
        return images_axial, images_sagital, images_coronal
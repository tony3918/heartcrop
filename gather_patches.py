import os
import numpy as np

'''data directory of heart ROI and CAC masks
structure: (patient_id) -> (image) and (label)'''
DATA_DIR = '/home/user/tony/DATA/np'

def crop_roi(image, roi_pos):
    '''image is 3D np.array (either CT or mask)
       roi_pos is list-like [x1,y1,z1,x2,y2,z2]'''
    x1,y1,z1,x2,y2,z2 = roi_pos
    return image[z1:z2,y1:y2,x1:x2]

def collect_patches_from_patient(image, label, w=25):
    '''image & label :  heart ROI and corresponding CAC mask (3D np.array)'''
    r = (w-1)//2
    patch_images = []
    patch_locs = []
    patch_labels = []
    for z in range(r, image.shape[0] - r):
        for y in range(r, image.shape[1] - r):
            for x in range(r, image.shape[2] - r):
                patch_images.append(image[z-r:z+r+1,y-r:y+r+1,x])
                patch_images.append(image[z-r:z+r+1,y,x-r:x+r+1])
                patch_images.append(image[z,y-r:y+r+1,x-r:x+r+1])
                for i in range(3):
                    patch_labels.append(label[z,y,x])
                    patch_locs.append([z/image.shape[0], y/image.shape[1], x/image.shape[2]])
    patch_images = np.array(patch_images)
    patch_locs = np.array(patch_locs)
    patch_labels = np.array(patch_labels)
    print(f'patch_images shape: {patch_images.shape}')
    print(f'patch_locs shape: {patch_locs.shape}')
    print(f'patch_labels shape: {patch_labels.shape}')
    return patch_images, patch_locs, patch_labels
    
def collect_patches(num_patients, train=True):
    if train:
        patient_list = sorted(os.listdir(DATA_DIR))[:num_patients]
    else:
        patient_list = sorted(os.listdir(DATA_DIR))[-num_patients:]
    
    images, locs, labels = [], [], [] 
    for patient in patient_list:
        image = np.load(os.path.join(DATA_DIR, patient, 'ct'))
        label = np.load(os.path.join(DATA_DIR, patient, 'mask'))
        patch_images, patch_locs, patch_labels = collect_patches_from_patient(image, label, w=25)
        images.append(patch_images)
        locs.append(patch_locs)
        labels.append(patch_labels)
    images = np.concatenate([patchset for patchset in images])
    locs = np.concatenate([patchset for patchset in locs])
    labels = np.concatenate([patchset for patchset in labels])
    print(f'images shape: {images.shape}')
    print(f'locs shape: {locs.shape}')
    print(f'labels shape: {labels.shape}')
    return images, locs, labels
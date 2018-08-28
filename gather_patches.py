import os
import numpy as np

'''data directory of heart ROI and CAC masks
structure: (patient_id) -> (image) and (label)'''
CT_DIR = '/home/user/tony/DATA/np/ct'
MASK_DIR = '/home/user/tony/DATA/np/mask'

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
        if z > 30:
            break
        for y in range(r, image.shape[1] - r):
            if y > 30:
                break
            for x in range(r, image.shape[2] - r):
                if x > 30:
                    break
                patch_images.append(image[z-r:z+r+1,y-r:y+r+1,x])
                patch_images.append(image[z-r:z+r+1,y,x-r:x+r+1])
                patch_images.append(image[z,y-r:y+r+1,x-r:x+r+1])
                for i in range(3):
                    patch_labels.append(label[z,y,x])
                    patch_locs.append([z/image.shape[0], y/image.shape[1], x/image.shape[2]])
    patch_images = np.array(patch_images)
    patch_locs = np.array(patch_locs)
    patch_labels = np.array(patch_labels)
    print(f'  patch_images shape: {patch_images.shape}')
    print(f'  patch_locs shape: {patch_locs.shape}')
    print(f'  patch_labels shape: {patch_labels.shape}')
    return patch_images, patch_locs, patch_labels
    
def collect_patches(num_patients, train=True):
    if train:
        patient_list = sorted(os.listdir(CT_DIR))[:num_patients]
    else:
        patient_list = sorted(os.listdir(CT_DIR))[-num_patients:]
    
    images, locs, labels = [], [], [] 
    for patient in patient_list:
        print(f'Collecting 2D patches from {patient}')
        image = np.load(os.path.join(CT_DIR, patient))
        label = np.load(os.path.join(MASK_DIR, patient))
        print(f'image: {image.shape}, label: {label.shape}')
        patch_images, patch_locs, patch_labels = collect_patches_from_patient(image, label, w=25)
        images.append(patch_images)
        locs.append(patch_locs)
        labels.append(patch_labels)
    images = np.concatenate([patchset for patchset in images]).astype(np.float32)
    locs = np.concatenate([patchset for patchset in locs]).astype(np.float32)
    labels = np.concatenate([patchset for patchset in labels]).astype(np.int32)
    print(f'images shape: {images.shape}')
    print(f'locs shape: {locs.shape}')
    print(f'labels shape: {labels.shape}')
    return images, locs, labels

if __name__ == '__main__':
    images, locs, labels = collect_patches(3, train=True)
    print(f'images: {images.dtype}')
    print(f'locs: {locs.dtype}')
    print(f'labels: {labels.dtype}')
import nibabel as nib
import numpy as np
import os
import scipy.ndimage

DATA_DIR = '/home/user/tony/DATA/raw'
SAVE_DIR = '/home/user/tony/DATA/np/mask'

def save_all_masks(DATA_DIR):
    '''save preprocessed masks for each patient in DATA_DIR'''
    for patient_id in os.listdir(DATA_DIR):
        if patient_id+'.npy' in os.listdir(SAVE_DIR):
            continue
        print(f'Saving mask for patient {patient_id}')
        filename = os.listdir(os.path.join(DATA_DIR, patient_id, 'CAC'))[0]
        mask = nib.load(os.path.join(DATA_DIR, patient_id, 'CAC', filename)).get_fdata()
        mask = np.transpose(mask)
        mask = mask[::-1,:,:]
        mask = preprocessed_mask(mask)
        np.save(os.path.join(SAVE_DIR, f'{patient_id}.npy'), mask)

def preprocessed_mask(original_mask):
    print(f'  original mask shape: {original_mask.shape}')
    mask = resample(original_mask, spacing=np.array([1.0,0.691406,0.691406]), new_spacing=[1.5,1.5,1.5])
    mask = np.where(mask>=128, 1, 0)
    print(f'  shape after resampling: {mask.shape}')
    mask = zeropad_crop(mask, size=256).astype(np.int8)
    print(f'  shape after zeropad_crop: {mask.shape}')
    return mask
    
def resample(ct, spacing, new_spacing = [1.5, 1.5, 1.5]):
    # resample ct to isotropic resolution (1.5mm x 1.5mm x 1.5mm)
    # considered edge cases due to rounding
    resize_factor = spacing / new_spacing
    new_shape = np.round(ct.shape * resize_factor)
    real_resize_factor = new_shape / ct.shape
    ct = scipy.ndimage.interpolation.zoom(ct, real_resize_factor)
    return ct

def zeropad_crop(ct, size=256):
    # 0-padding and/or cropping along outer image borders to match 256x256x256
    
    # processing for z-axis
    if ct.shape[0] < size:
        ct = np.insert(ct, 0, np.zeros([round((size-ct.shape[0]+1)/2),ct.shape[1],ct.shape[1]]), axis=0)
        ct = np.append(ct, np.zeros([size-ct.shape[0],ct.shape[1],ct.shape[1]]), axis=0)
    elif ct.shape[0] > size:
        idx = round((ct.shape[0]-size)/2)
        ct = ct[idx:idx+size,:,:]
    
    # processing for x and y axis (assuming x,y dimensions are equal)
    if ct.shape[1] < size:
        idx = round((size-ct.shape[1])/2)
        tmp = np.zeros([size,size,size])
        tmp[:,idx:idx+ct.shape[1],idx:idx+ct.shape[1]] = ct
        ct = tmp
    elif ct.shape[1] > size:
        idx = round((ct.shape[1]-size)/2)
        ct = ct[:,idx:idx+size,idx:idx+size]
    
    return ct


def windowing(ct, window_width=350, window_level=50):
    # apply window level & width, zero center pixel values and convert dtype to uint8 (-128~127)
    ct = ct*(255/(2*window_width)) - (window_level-window_width+1024)*255/(2*window_width)
    ct[ct < 0] = 0
    ct[ct > 255] = 255
    ct = ct - 128
    ct = ct.astype(np.int8)
    return ct


if __name__ == '__main__':
    print(f"Saving all preprocessed masks for each patient in {DATA_DIR}")
    save_all_masks(DATA_DIR)
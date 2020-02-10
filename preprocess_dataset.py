#!/usr/bin/env python
# coding: utf-8

# # Arrange Dataset

# In[1]:


from __future__ import print_function

import os
from natsort import natsorted
import pandas as pd
import numpy as np
import nrrd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
import sys, time, os
import random
import string
import pydicom
from pydicom.data import get_testdata_files


# In[2]:


def read_nrrd(filename, i):
    # Read the data back from file
    readdata, header = nrrd.read(filename)
    
    img = readdata[:, : , i]
    rows, cols = img.shape[:2]

    # rotate 270 degree
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1.0)

    img = cv2.warpAffine(img, M,(cols, rows))

    normalized = cv2.normalize(img,
                               None,
                               0, 255,
                               cv2.NORM_MINMAX,
                               cv2.CV_8UC1)
    
    return normalized


# In[3]:


def get_segment_crop(img,tol=0, mask=None):
    img = cv2.bitwise_and(img, img, mask=mask)    
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


# In[4]:


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


# In[5]:


def writeSlices(save_dir, series_tag_values, new_img, i):
    image_slice = new_img[:,:,i]

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
    
    image_slice.SetMetaData("0010|0020", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    dcm_path = os.path.join(save_dir,str(i)+'.dcm')
    writer.SetFileName(dcm_path)
    writer.Execute(image_slice)
    
    ds = pydicom.dcmread(dcm_path)
    ds.PatientID = randomString()
    ds.save_as(dcm_path)

# In[6]:


src_data_root = '/nasdata/PET_dataset/DSMC_breast370'


# In[7]:


annotation_file = os.path.join(src_data_root, 'LN meta.xlsx')
image_root = os.path.join(src_data_root, 'Breast_370')

anno_data = pd.read_excel(annotation_file)


# In[8]:


for proc_type in ['mask', 'full']:
    target_data_root = os.path.join('/data/DSMC_breast370',proc_type)

    for e in ['0', '1']:
        os.system(f'mkdir -p {os.path.join(target_data_root, e)}')

    for lab, case in tqdm(anno_data.values, desc=proc_type):
        target_dir = os.path.join(target_data_root, str(lab))
        target_filedir = os.path.join(target_dir, f'Case_{case}')
        dcm_savedir = os.path.join(target_filedir, 'DCMs')
        
        os.system(f'mkdir -p {target_filedir}')
        os.system(f'mkdir -p {dcm_savedir}')
        
        raw_filename = os.path.join(image_root,
                                    f'Case ({case})',
                                    'img.nrrd')
        mask_filename = os.path.join(image_root,
                                    f'Case ({case})',
                                    'lab.nrrd')
        
        readdata, header = nrrd.read(raw_filename)
        depth = readdata.shape[-1]

        res = []
        for i in range(depth):
            raw_img = read_nrrd(raw_filename, i)
            mask_img = read_nrrd(mask_filename, i)
            if proc_type == 'mask':
                raw_img = cv2.bitwise_and(raw_img, raw_img, mask=mask_img)
            elif proc_type == 'mask_crop':
                raw_img = get_segment_crop(raw_img, mask=mask_img)

            # write raw_img as .PNG file
            target_filename = os.path.join(target_filedir,
                                           f'{i}.png')
            cv2.imwrite(target_filename, raw_img)
            
            res.append(raw_img)
            
        # write raw_img as .dcm file
        # Create a new series from a numpy array
        new_arr = np.array(res, dtype=np.int16)
        new_img = sitk.GetImageFromArray(new_arr)
        # new_img.SetSpacing([2.5,3.5,4.5])

        # Write the 3D image as a series
        # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
        #            original image. This is a delicate opration and requires knowlege of
        #            the DICOM standard. This example only modifies some. For a more complete
        #            list of tags that need to be modified see:
        #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
        #            If it is critical for your work to generate valid DICOM files,
        #            It is recommended to use David Clunie's Dicom3tools to validate the files 
        #                           (http://www.dclunie.com/dicom3tools.html).

        writer = sitk.ImageFileWriter()
        # Use the study/series/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        # Copy some of the tags and add the relevant tags indicating the change.
        # For the series instance UID (0020|000e), each of the components is a number, cannot start
        # with zero, and separated by a '.' We create a unique series ID using the date and time.
        # tags of interest:
        direction = new_img.GetDirection()
        series_tag_values = [("0008|0031",modification_time), # Series Time
                          ("0008|0021",modification_date), # Series Date
                          ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                          ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                          ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                            direction[1],direction[4],direction[7])))),
                          ("0008|103e", "Created-SimpleITK")] # Series Description

        # Write slices to output directory
        list(map(lambda i: writeSlices(dcm_savedir, series_tag_values, new_img, i), range(new_img.GetDepth())))


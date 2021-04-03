import os
import shutil
import numpy as np
from copy import deepcopy as copy
import cv2
import math
import datetime
import time
from scipy import ndimage as ndi
from scipy.io import loadmat
import numpy as np
import scipy
from scipy.ndimage.interpolation import zoom
from glob import glob
from skimage import measure, morphology
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure,distance_transform_edt
from skimage.morphology import convex_hull_image,disk,binary_closing
from multiprocessing import Pool
from functools import partial
import sys
sys.path.append('preprocessing')
from step1b import step1_python
import warnings
import pydicom as dicom


def load_scan(path):
	"""
	:param path: input CT path, dicom format
	:return: CT image
	"""
	#slices = [dicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)]
	slices = []
	for s in os.listdir(path):
		if s.endswith('.dcm'):
			slices.append(dicom.read_file(path + '/' + s, force=True))

	slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
	if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
		sec_num = 2
		while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
			sec_num = sec_num + 1
		slice_num = int(len(slices) / sec_num)
		slices.sort(key=lambda x:float(x.InstanceNumber))
		slices = slices[0:slice_num]
		slices.sort(key=lambda x:float(x.ImagePositionPatient[2]))
	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
		
	for s in slices:
		s.SliceThickness = slice_thickness
		
	return slices


def resample(imgs, spacing, new_spacing, order=1):
	"""
	:param imgs: CT image
	:param spacing: CT original voxel spacing
	:param new_spacing: target voxel spacing
	:param order: interpolation choice
	:return: CT image with new voxel spacing, new voxel spacing
	"""
	if len(imgs.shape) == 3:
		new_shape = np.round(imgs.shape * spacing / new_spacing)
		true_spacing = spacing * imgs.shape / new_shape
		resize_factor = new_shape / imgs.shape
		imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
		return imgs, true_spacing
	elif len(imgs.shape) == 4:
		n = imgs.shape[-1]
		newimg = []
		for i in range(n):
			slice = imgs[:,:,:,i]
			newslice, true_spacing = resample(slice, spacing, new_spacing)
			newimg.append(newslice)
		newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
		return newimg, true_spacing
	else:
		raise ValueError('wrong shape')


def process_mask(mask):
	"""
	:param mask: input lung mask
	:return: convex hull processing on lung mask to avoid over-segmentation
	"""
	convex_mask = np.copy(mask)
	for i_layer in range(convex_mask.shape[0]):
		mask1 = np.ascontiguousarray(mask[i_layer])
		if np.sum(mask1) > 0:
			mask2 = convex_hull_image(mask1)
			if np.sum(mask2) > 1.5*np.sum(mask1):
				mask2 = mask1
		else:
			mask2 = mask1
		convex_mask[i_layer] = mask2
	struct = generate_binary_structure(3, 1)
	dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
	return dilatedMask

def lumTrans_hu(img):
	"""
	:param img: CT image
	:return: Hounsfield Unit window clipped and normalized
	"""
	lungwin = np.array([-1000.,400.])
	newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	newimg = (newimg*255).astype('uint8')
	return newimg

def save_itk(image, origin, spacing, filename):
	"""
	:param image: images to be saved
	:param origin: CT origin
	:param spacing: CT spacing
	:param filename: save name
	:return: None
	"""
	if type(origin) != tuple:
		if type(origin) == list:
			origin = tuple(reversed(origin))
		else:
			origin = tuple(reversed(origin.tolist()))
	if type(spacing) != tuple:
		if type(spacing) == list:
			spacing = tuple(reversed(spacing))
		else:
			spacing = tuple(reversed(spacing.tolist()))
	itkimage = sitk.GetImageFromArray(image, isVector=False)
	itkimage.SetSpacing(spacing)
	itkimage.SetOrigin(origin)
	sitk.WriteImage(itkimage, filename, True)

def savenpy(data_path, prep_folder):
	"""
	:param data_path: input CT data path
	:param prep_folder:
	:return: None
	"""
	resolution = np.array([1, 1, 1])
	name = data_path.split('/')[-1].split('.nii')[0]
	assert (os.path.exists(data_path) is True)
	im, m1, m2, mtotal, origin, spacing = step1_python(data_path)
	# print ('Origin: ', origin, ' Spacing: ', spacing, 'img shape: ', im.shape)
	Mask = m1 + m2
	xx, yy, zz = np.where(Mask)
	box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz),np.max(zz)]])
	margin = 5
	box = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([np.array(Mask.shape), box[:, 1] + margin], axis=0).T]).T

	# save the lung mask
	data_savepath = os.path.join(prep_folder, name+'_lung_mask.nii.gz')
	Mask_crop = Mask[box[0, 0]:box[0, 1],
				  box[1,0]:box[1,1],
				  box[2,0]:box[2,1]]
	save_itk(Mask_crop.astype(dtype='uint8'), origin, spacing, data_savepath)
	# convex_mask = m1
	# dm1 = process_mask(m1)
	# dm2 = process_mask(m2)
	# dilatedMask = (dm1 + dm2) | mtotal
	# Mask = m1+m2
	# dilatedMask = dilatedMask.astype('uint8')
	# Mask = Mask.astype('uint8')

	# save the CT image
	im[np.isnan(im)] = -2000
	sliceim_hu = lumTrans_hu(im)
	shapeorg = sliceim_hu.shape
	box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0, shapeorg[2]]])
	sliceim2_hu = sliceim_hu[box[0, 0]:box[0, 1],
				  box[1,0]:box[1,1],
				  box[2,0]:box[2,1]]
	# save box (image original shape and cropped window region)
	box = np.concatenate([box, box_shape], axis=0)
	np.save(os.path.join(prep_folder, name+'_box.npy'), box)
	# save processed image
	data_savepath = os.path.join(prep_folder, name+'_clean_hu.nii.gz')
	save_itk(sliceim2_hu.astype(dtype='uint8'), origin, spacing, data_savepath)
	
	return

def preprocess_CT(inputpath=None, savepath=None):
	"""
	Preprocess the CT images in the input path to extract lung field
	Save the processed images in the save path
	:param inputpath: input data path
	:param savepath: output save path
	:return: save directory path
	"""
	warnings.filterwarnings("ignore")
	warnings.simplefilter(action='ignore', category=FutureWarning)
	if not os.path.exists(savepath):
		os.mkdir(savepath)

	filelist = glob(os.path.join(inputpath, '*.nii*'))  # default nifty format
	print (inputpath, filelist)
	for curfilepath in filelist:
		start_time = time.time()
		print ('starting preprocessing lung CT')
		savenpy(data_path=curfilepath, prep_folder=savepath)
		end_time = time.time()
		print ('end preprocessing lung CT, time %d seconds'%(end_time-start_time))

	return savepath

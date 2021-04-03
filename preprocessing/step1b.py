import numpy as np # linear algebra
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure, morphology
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import pydicom as dicom

def load_scan(path):
	#slices = [dicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)]
	slices = []
	for s in os.listdir(path):
		if s.endswith('.dcm'):
			slices.append(dicom.read_file(path + '/' + s, force = True))
	
	assert (len(slices) > 0)
	slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
	if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
		sec_num = 2;
		while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
			sec_num = sec_num+1;
		slice_num = int(len(slices) / sec_num)
		slices.sort(key = lambda x:float(x.InstanceNumber))
		slices = slices[0:slice_num]
		slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
		
	for s in slices:
		s.SliceThickness = slice_thickness
		
	return slices

def get_pixels_hu(slices):
	image = np.stack([s.pixel_array for s in slices])
	# Convert to int16 (from sometimes int16), 
	# should be possible as values should always be low enough (<32k)
	image = image.astype(np.int16)
	# Convert to Hounsfield units (HU)
	for slice_number in range(len(slices)):        
		intercept = slices[slice_number].RescaleIntercept
		slope = slices[slice_number].RescaleSlope
		
		if slope != 1:
			image[slice_number] = slope * image[slice_number].astype(np.float64)
			image[slice_number] = image[slice_number].astype(np.int16)
			
		image[slice_number] += np.int16(intercept)
	origin = slices[0].ImagePositionPatient
	origin_str = [float(origin[0].original_string), float(origin[1].original_string), float(origin[2].original_string)]
	#print (type(origin))
	zspacing = slices[0].SliceThickness
	#print (zspacing)
	xyspacing = slices[0].PixelSpacing
	#print (type(xyspacing[0]))
	#assert (1>2)
	spacing = [float(zspacing), float(xyspacing[0]), float(xyspacing[1])]
	#print (origin_str, spacing)
	return np.array(image, dtype=np.int16), np.array(origin_str, dtype=np.float32), np.array(spacing, dtype=np.float32)


def binarize_per_slice(image, spacing, intensity_th=-300, sigma=1, area_th=20, eccen_th=0.99, bg_patch_size=10):
	bw = np.zeros(image.shape, dtype=bool)
	
	# prepare a mask, with all corner values set to nan
	image_size = image.shape[1]
	grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
	x, y = np.meshgrid(grid_axis, grid_axis)
	d = (x**2+y**2)**0.5
	nan_mask = (d<image_size/2).astype(float)
	nan_mask[nan_mask == 0] = np.nan
	for i in range(image.shape[0]):
		# Check if corner pixels are identical, if so the slice  before Gaussian filtering
		if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1 or\
		len(np.unique(image[i, -bg_patch_size:, -bg_patch_size:])) == 1 or\
		len(np.unique(image[i, 0:bg_patch_size, -bg_patch_size:])) == 1 or\
		len(np.unique(image[i, -bg_patch_size:, 0:bg_patch_size])) == 1:
			current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
		else:
			current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
		
		# select proper components
		label = measure.label(current_bw)
		properties = measure.regionprops(label)
		valid_label = set()
		for prop in properties:
			if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
				valid_label.add(prop.label)
		current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
		bw[i] = current_bw
		
	return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 7.5], area_th=2e3, dist_th=50):
	# in some cases, several top layers need to be removed first
	if cut_num > 0:
		bw0 = np.copy(bw)
		bw[-cut_num:] = False
	
	label = measure.label(bw, connectivity=1)
	# remove components access to corners
	mid = int(label.shape[2] / 2)
	bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
					label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
					label[0, 0, mid], label[0, -1, mid], label[0, mid, 0], label[0, mid, -1],\
					label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid], label[-1-cut_num, mid, 0], label[-1-cut_num, mid, -1]])
	for l in bg_label:
		label[label == l] = 0

	# select components based on volume
	properties = measure.regionprops(label)
	for prop in properties:
		if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
			label[label == prop.label] = 0
			
	# prepare a distance map for further analysis
	x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
	y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
	x, y = np.meshgrid(x_axis, y_axis)
	d = (x**2+y**2)**0.5
	vols = measure.regionprops(label)
	valid_label = set()
	# select components based on their area and distance to center axis on all slices
	for vol in vols:
		single_vol = label == vol.label
		slice_area = np.zeros(label.shape[0])
		min_distance = np.zeros(label.shape[0])
		for i in range(label.shape[0]):
			slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
			min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
		
		if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
			valid_label.add(vol.label)
			
	bw = np.in1d(label, list(valid_label)).reshape(label.shape)
	# fill back the parts removed earlier
	if cut_num > 0:
		# bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
		bw1 = np.copy(bw)
		bw1[-cut_num:] = bw0[-cut_num:]
		bw2 = np.copy(bw)
		bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
		bw3 = bw1 & bw2
		label = measure.label(bw, connectivity=1)
		label3 = measure.label(bw3, connectivity=1)
		l_list = list(set(np.unique(label)) - {0})
		valid_l3 = set()
		for l in l_list:
			indices = np.nonzero(label==l)
			l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
			if l3 > 0:
				valid_l3.add(l3)
		bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
	
	return bw, len(valid_label)

def fill_hole(bw):
	# fill 3d holes
	label = measure.label(~bw)
	# idendify corner components
	mid = int(label.shape[2] / 2)
	bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
					label[0, 0, mid], label[0, -1, mid], label[0, mid, 0], label[0, mid, -1],\
					label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1],\
					label[-1, 0, mid], label[-1, -1, mid], label[-1, mid, 0], label[-1, mid, -1]])
	bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
	return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
	def extract_main(bw, cover=0.95):
		for i in range(bw.shape[0]):
			current_slice = bw[i]
			label = measure.label(current_slice)
			properties = measure.regionprops(label)
			properties.sort(key=lambda x: x.area, reverse=True)
			area = [prop.area for prop in properties]
			count = 0
			sum = 0
			while sum < np.sum(area)*cover:
				sum = sum+area[count]
				count = count+1
			filter = np.zeros(current_slice.shape, dtype=bool)
			for j in range(count):
				bb = properties[j].bbox
				filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
			bw[i] = bw[i] & filter
		   
		label = measure.label(bw)
		properties = measure.regionprops(label)
		properties.sort(key=lambda x: x.area, reverse=True)
		bw = label==properties[0].label

		return bw
	
	def fill_2d_hole(bw):
		for i in range(bw.shape[0]):
			current_slice = bw[i]
			label = measure.label(current_slice)
			properties = measure.regionprops(label)
			for prop in properties:
				bb = prop.bbox
				current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
			bw[i] = current_slice

		return bw
	
	found_flag = False
	iter_count = 0
	bw0 = np.copy(bw)
	while not found_flag and iter_count < max_iter:
		label = measure.label(bw, connectivity=2)
		properties = measure.regionprops(label)
		properties.sort(key=lambda x: x.area, reverse=True)
		if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
			found_flag = True
			bw1 = label == properties[0].label
			bw2 = label == properties[1].label
		else:
			bw = scipy.ndimage.binary_erosion(bw)
			iter_count = iter_count + 1
	
	if found_flag:
		d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
		d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
		bw1 = bw0 & (d1 < d2)
		bw2 = bw0 & (d1 > d2)
				
		bw1 = extract_main(bw1)
		bw2 = extract_main(bw2)
		
	else:
		bw1 = bw0
		bw2 = np.zeros(bw.shape).astype('bool')
		
	bw1 = fill_2d_hole(bw1)
	bw2 = fill_2d_hole(bw2)
	bw = bw1 | bw2

	return bw1, bw2, bw

def load_itk_image(filename):
	itkimage = sitk.ReadImage(filename)
	numpyImage = sitk.GetArrayFromImage(itkimage)
	numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
	numpySpacing = np.array(list(reversed(itkimage.GetSpacing()))) 
	return numpyImage, numpyOrigin, numpySpacing

def save_itk(image, origin, spacing, filename):
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

def dilation_mask(bw):
	struct = generate_binary_structure(2,1)  
	for islice in range(bw.shape[0]):
		curslice = bw[islice]
		##print ('curslice shape: ', curslice.shape)
		bw[islice] = binary_dilation(curslice,structure=struct,iterations=2)
	return bw

def step1_python(case_path):
	#case = load_scan(case_path)
	#case_pixels, origin, spacing = get_pixels_hu(case)
	case_pixels, origin, spacing = load_itk_image(case_path)
	shape_original = case_pixels.shape
	start_id_1 = (512 - shape_original[1])//2
	start_id_2 = (512 - shape_original[2])//2

	if not shape_original[1] == shape_original[2]:
		new_case_pixels = np.ones([shape_original[0],512,512]) * (-1000) # HU

		new_case_pixels[:,start_id_1:shape_original[1]+start_id_1,start_id_2:start_id_2+shape_original[2]] = case_pixels
		case_pixels = new_case_pixels

	bw = binarize_per_slice(case_pixels, spacing)
	# bw_test = bw[:,start_id_1:start_id_1+shape_original[1],start_id_2:start_id_2+shape_original[2]]
	# save_itk(bw_test.astype('uint8'), origin, spacing, '/run/media/qin/yolayDATA2/CARVE14/data/lungs_correction/bw_1.nii.gz')
	flag = 0
	cut_num = 0
	cut_step = 3
	bw0 = np.copy(bw)
	while flag == 0 and cut_num < bw.shape[0]:
		bw = np.copy(bw0)
		bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
		if flag != 0:
			break
		else:
			cut_num = cut_num + cut_step
	bw = fill_hole(bw)
	#bw = ~bw

	#bw = dilation_mask(bw)
	bw_copy = bw.copy()
	bw1, bw2, bw = two_lung_only(bw, spacing)
	
	
	if not shape_original[1] == shape_original[2]:
		case_pixels = case_pixels[:,start_id_1:start_id_1+shape_original[1],start_id_2:start_id_2+shape_original[2]]
		bw1 = bw1[:,start_id_1:start_id_1+shape_original[1],start_id_2:start_id_2+shape_original[2]]
		bw2 = bw2[:,start_id_1:start_id_1+shape_original[1],start_id_2:start_id_2+shape_original[2]]
		bw_copy = bw_copy[:,start_id_1:start_id_1+shape_original[1],start_id_2:start_id_2+shape_original[2]]

	return case_pixels, bw1, bw2, bw_copy, origin, spacing

def load_pixels(case_path):
	case = load_scan(case_path)
	case_pixels, origin, spacing = get_pixels_hu(case)
	return case_pixels, origin, spacing

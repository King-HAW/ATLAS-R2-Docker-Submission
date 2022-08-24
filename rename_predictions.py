import os
import scipy
from settings import loader_settings
import numpy as np
import SimpleITK as sitk
import shutil

from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
from typing import Union, Tuple
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg


def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):

    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if region_class_order is None:
        fg_softmax = seg_old_spacing[1]
        fg_ones_thres_1 = np.zeros(fg_softmax.shape)
        fg_ones_thres_2 = np.zeros(fg_softmax.shape)
        tmp_thresh = np.max(fg_softmax)*0.7
        num_fg = np.count_nonzero(fg_softmax>0.5)
        tmp_big = 0.55
        tmp_small = 0.45
        num_limit = True
        if num_fg<3000:
                tmp_big = 0.7
                tmp_small = 0.5
        else:
                tmp_big = 0.55
                tmp_small = 0.5
                num_limit = False
        #tmp_thresh #min(tmp_thresh,0.6)
        tmp_second = min(tmp_thresh,0.55)
        fg_ones_thres_1[fg_softmax > tmp_big] = 1 #get_thresh1_compnent(fg_softmax,0.6,0.5)#
        #fg_ones_thres_2[fg_softmax > 0.4] = 1
        fg_ones_thres_2[fg_softmax > tmp_small] = 1

        labeled_fg, num_lesions_fg = scipy.ndimage.label(fg_ones_thres_1.astype(bool))

        if num_lesions_fg >4 and num_limit:
            component_size_dict = {}
            for idx_lesion in range(1, num_lesions_fg+1):
                lesion_thres_component = labeled_fg == idx_lesion
                component_size_dict[idx_lesion] = np.sum(lesion_thres_component)
            component_size_list_sorted = sorted(component_size_dict.items(), key=lambda kv:kv[1])
            mean_prob = np.Inf
            used_idx = 0
            for idx in range(3): #search the block with smallest mean from min Top3 area blocks 
                component_idx = component_size_list_sorted[idx][0]
                lesion_thres_component = labeled_fg == component_idx
                if np.sum(fg_softmax * lesion_thres_component)/ np.sum(lesion_thres_component) < mean_prob: 
                    mean_prob = np.mean(fg_softmax * lesion_thres_component)
                    used_idx = component_idx
            
            component_need_delete = labeled_fg == used_idx
            fg_ones_thres_1 = fg_ones_thres_1 - component_need_delete
            labeled_fg, num_lesions_fg = scipy.ndimage.label(fg_ones_thres_1.astype(bool))

        add_unlimit_blocks = np.zeros(fg_softmax.shape)
        if num_lesions_fg >=0:
            if np.sum(fg_ones_thres_1) == 0:
                tmp_thresh = np.max(fg_softmax) - 0.1
                fg_ones_thres_tmp = np.zeros(fg_softmax.shape)
                fg_ones_thres_tmp[fg_softmax > tmp_thresh] = 1
                seg_old_spacing = fg_ones_thres_tmp

            else:
                fg_diff = fg_ones_thres_2 - fg_ones_thres_1
                if np.sum(fg_diff) == 0:
                    fg_ones_thres_2 = np.zeros(fg_softmax.shape)
                    fg_ones_thres_2[fg_softmax > 0.3] = 1
                    fg_diff = fg_ones_thres_2 - fg_ones_thres_1

                #labeled_fg_diff, num_lesions = scipy.ndimage.label(fg_diff.astype(bool))
                labeled_fg_2, num_lesions = scipy.ndimage.label(fg_ones_thres_2.astype(bool))
                # if verbose: print("The number of connected-components in diff: {}".format(num_lesions))

                find_point_flag = False
                mark_one_voxel = None
                max_area = 0
                seg_add_spacing =  np.zeros(fg_softmax.shape)
                for idx_lesion in range(1, num_lesions+1):
                    #lesion_diff_thres_component = labeled_fg_diff == idx_lesion
                    lesion_thres2_component = labeled_fg_2 == idx_lesion
                    intersection = np.zeros(fg_softmax.shape)
                    intersection = lesion_thres2_component*labeled_fg
                    num_intersection = len(np.unique(intersection))
                    if  num_intersection>= 2:
                        if num_intersection==2 or num_limit:
                            seg_add_spacing = np.maximum(seg_add_spacing,lesion_thres2_component)
                        continue
                    #continue
                    if num_limit:
                        continue
                    add_unlimit_blocks += lesion_thres2_component
                    weight_area = np.sum(lesion_thres2_component*fg_softmax)

                    max_voxel_value = np.max(fg_softmax*lesion_thres2_component)

                    if weight_area<max_area:
                        continue
                    max_area = weight_area
                    #fg_add_one_voxel = np.zeros(fg_softmax.shape)
                    #x,y,z = np.nonzero_idx(lesion_thres2_component)
                    
                    #x, y, z = np.where(fg_softmax*lesion_thres2_component==max_voxel_value)
                    #fg_add_one_voxel[list(zip(x, y, z))[0]] = 1
                    fg_add_one_voxel = np.zeros(fg_softmax.shape)
                    fg_add_one_voxel[lesion_thres2_component>0] = 1
                    find_point_flag = True

                if not num_limit:
                    fg_add_one_voxel = add_unlimit_blocks
                if not find_point_flag:
                    seg_old_spacing = fg_ones_thres_1
                else:
                    seg_old_spacing = fg_ones_thres_1 + fg_add_one_voxel
                seg_old_spacing = np.maximum(seg_old_spacing, seg_add_spacing)

    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.7] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)


if __name__ == '__main__':
    nnunet_prediction_path= '/opt/algorithm/ensemble/predictions'
    inp_path = loader_settings['InputPath']  # Path for the default input
    out_path = loader_settings['OutputPath']  # Path for the default output

    Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)

    file_name_list = os.listdir(inp_path)

    # determine whether to use postprocessing
    use_postprocessing_flag = True

    if use_postprocessing_flag:
        for file_name in file_name_list:
            base_file_name = file_name.split('.')[0]
            prediction_nii_path = os.path.join(nnunet_prediction_path, base_file_name + '.nii.gz')
            prediction_pkl_path = os.path.join(nnunet_prediction_path, base_file_name + '.pkl')
            prediction_npz_path = os.path.join(nnunet_prediction_path, base_file_name + '.npz')

            softmax = np.load(prediction_npz_path)['softmax']
            props = load_pickle(prediction_pkl_path)
            out_file = os.path.join(out_path, file_name)
            regions_class_order = None

            save_segmentation_nifti_from_softmax(softmax, out_file, props[0], 3, regions_class_order, None, None, force_separate_z=None)

    else:
        for file_name in file_name_list:
            if '.nii.gz' in file_name: # suffix is .nii.gz
                shutil.copyfile(os.path.join(nnunet_prediction_path, file_name), os.path.join(out_path, file_name))

            else: # suffix is not .nii.gz
                base_file_name = file_name.split('.')[0]
                prediction_sitk_img = sitk.ReadImage(os.path.join(nnunet_prediction_path, base_file_name + '.nii.gz'))
                sitk.WriteImage(prediction_sitk_img, os.path.join(out_path, file_name))

    print("Done")
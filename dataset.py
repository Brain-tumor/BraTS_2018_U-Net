import random

import gc
import os
import pickle

import nibabel as nib
import numpy as np
import tensorlayer as tl

"""
In seg file
--------------
Label 1: necrotic and non-enhancing tumor
Label 2: edema 
Label 4: enhancing tumor
Label 0: background

MRI
-------
whole/complete tumor: 1 2 4
core: 1 4
enhance: 4
"""
###============================= SETTINGS ===================================###
DATA_SIZE = 'small'  # (small, half or all)

save_dir = "data/train_dev_all/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

HGG_data_path = "data/MICCAI_BraTS_2018_Data_Training/HGG"
LGG_data_path = "data/MICCAI_BraTS_2018_Data_Training/LGG"
survival_csv_path = "data/MICCAI_BraTS_2018_Data_Training/survival_data.csv"
###==========================================================================###


class Dataset:
    def __init__(self):
        self.HGG_name_list = []
        self.LGG_name_list = []
        self.data_types = ['flair', 't1', 't1ce', 't2']
        self.HGG_name_train = []
        self.HGG_name_dev = []
        self.HGG_name_test = []
        self.LGG_name_train = []
        self.LGG_name_dev = []
        self.LGG_name_test = []

        self.X_train_input = []
        self.X_train_target = []

        self.X_dev_input = []
        self.X_dev_target = []

        self.load_data_list()
        self.get_data_split()
        self.load_image()

    def load_data_list(self):
        HGG_path_list = []
        LGG_path_list = []
        if DATA_SIZE == 'all':
            HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
            LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)
        elif DATA_SIZE == 'half':
            HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:100]  # DEBUG WITH SMALL DATA
            LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:30]  # DEBUG WITH SMALL DATA
        elif DATA_SIZE == 'small':
            HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:50]  # DEBUG WITH SMALL DATA
            LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:20]  # DEBUG WITH SMALL DATA
        else:
            exit("Unknown DATA_SIZE")
        print(len(HGG_path_list), len(LGG_path_list))  # 210 #75

        self.HGG_name_list = [os.path.basename(p) for p in HGG_path_list]
        self.LGG_name_list = [os.path.basename(p) for p in LGG_path_list]

    def get_data_split(self):
        dev_index_LGG = []
        test_index_LGG = []
        tr_index_LGG = []
        dev_index_HGG = []
        test_index_HGG = []
        tr_index_HGG = []

        index_HGG = list(range(0, len(self.HGG_name_list)))
        index_LGG = list(range(0, len(self.LGG_name_list)))
        random.shuffle(index_HGG)
        random.shuffle(index_HGG)

        if DATA_SIZE == 'all':
            dev_index_HGG = index_HGG[-84:-42]
            test_index_HGG = index_HGG[-42:]
            tr_index_HGG = index_HGG[:-84]
            dev_index_LGG = index_LGG[-30:-15]
            test_index_LGG = index_LGG[-15:]
            tr_index_LGG = index_LGG[:-30]
        elif DATA_SIZE == 'half':
            dev_index_HGG = index_HGG[-30:]  # DEBUG WITH SMALL DATA
            test_index_HGG = index_HGG[-5:]
            tr_index_HGG = index_HGG[:-30]
            dev_index_LGG = index_LGG[-10:]  # DEBUG WITH SMALL DATA
            test_index_LGG = index_LGG[-5:]
            tr_index_LGG = index_LGG[:-10]
        elif DATA_SIZE == 'small':
            dev_index_HGG = index_HGG[35:42]  # DEBUG WITH SMALL DATA
            # print(index_HGG, dev_index_HGG)
            # exit()
            test_index_HGG = index_HGG[41:42]
            tr_index_HGG = index_HGG[0:35]
            dev_index_LGG = index_LGG[7:10]  # DEBUG WITH SMALL DATA
            test_index_LGG = index_LGG[9:10]
            tr_index_LGG = index_LGG[0:7]

        self.HGG_name_dev = [self.HGG_name_list[i] for i in dev_index_HGG]
        self.HGG_name_test = [self.HGG_name_list[i] for i in test_index_HGG]
        self.HGG_name_train = [self.HGG_name_list[i] for i in tr_index_HGG]

        self.LGG_name_dev = [self.LGG_name_list[i] for i in dev_index_LGG]
        self.LGG_name_test = [self.LGG_name_list[i] for i in test_index_LGG]
        self.LGG_name_train = [self.LGG_name_list[i] for i in tr_index_LGG]

    def image_normalization(self):

        data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in self.data_types}

        # calculate mean and std for all data types

        # preserving_ratio = 0.0
        # preserving_ratio = 0.01 # 0.118 removed
        # preserving_ratio = 0.05 # 0.213 removed
        # preserving_ratio = 0.10 # 0.359 removed

        # ==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD
        data_temp_list = []
        for i in self.data_types:
            data_temp_list = []
            for j in self.HGG_name_list:
                img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii.gz')
                img = nib.load(img_path).get_data()
                data_temp_list.append(img)

            for j in self.LGG_name_list:
                img_path = os.path.join(LGG_data_path, j, j + '_' + i + '.nii.gz')
                img = nib.load(img_path).get_data()
                data_temp_list.append(img)

            data_temp_list = np.asarray(data_temp_list)
            m = np.mean(data_temp_list)
            s = np.std(data_temp_list)
            data_types_mean_std_dict[i]['mean'] = m
            data_types_mean_std_dict[i]['std'] = s
        del data_temp_list
        print(data_types_mean_std_dict)

        with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
            pickle.dump(data_types_mean_std_dict, f, protocol=4)
        return data_types_mean_std_dict

    def load_image(self):
        data_types_mean_std_dict = self.image_normalization()

        ##==================== GET NORMALIZE IMAGES

        print(" HGG Validation")
        for i in self.HGG_name_dev:
            all_3d_data = []
            for j in self.data_types:
                img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_data()
                img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_3d_data.append(img)

            seg_path = os.path.join(HGG_data_path, i, i + '_seg.nii.gz')
            seg_img = nib.load(seg_path).get_data()
            seg_img = np.transpose(seg_img, (1, 0, 2))
            for j in range(all_3d_data[0].shape[2]):
                combined_array = np.stack(
                    (all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]),
                    axis=2)
                combined_array = np.transpose(combined_array, (1, 0, 2))  # .tolist()
                combined_array.astype(np.float32)
                self.X_dev_input.append(combined_array)

                seg_2d = seg_img[:, :, j]
                seg_2d.astype(int)
                self.X_dev_target.append(seg_2d)
            del all_3d_data
            gc.collect()
            print("finished {}".format(i))

        print(" LGG Validation")
        for i in self.LGG_name_dev:
            all_3d_data = []
            for j in self.data_types:
                img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_data()
                img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_3d_data.append(img)

            seg_path = os.path.join(LGG_data_path, i, i + '_seg.nii.gz')
            seg_img = nib.load(seg_path).get_data()
            seg_img = np.transpose(seg_img, (1, 0, 2))
            for j in range(all_3d_data[0].shape[2]):
                combined_array = np.stack(
                    (all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]),
                    axis=2)
                combined_array = np.transpose(combined_array, (1, 0, 2))  # .tolist()
                combined_array.astype(np.float32)
                self.X_dev_input.append(combined_array)

                seg_2d = seg_img[:, :, j]
                seg_2d.astype(int)
                self.X_dev_target.append(seg_2d)
            del all_3d_data
            gc.collect()
            print("finished {}".format(i))

        self.X_dev_input = np.asarray(self.X_dev_input, dtype=np.float32)
        self.X_dev_target = np.asarray(self.X_dev_target)

        print(" HGG Train")
        for i in self.HGG_name_train:
            all_3d_data = []
            for j in self.data_types:
                img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_data()
                img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_3d_data.append(img)

            seg_path = os.path.join(HGG_data_path, i, i + '_seg.nii.gz')
            seg_img = nib.load(seg_path).get_data()
            seg_img = np.transpose(seg_img, (1, 0, 2))
            for j in range(all_3d_data[0].shape[2]):
                combined_array = np.stack(
                    (all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]),
                    axis=2)
                combined_array = np.transpose(combined_array, (1, 0, 2))  # .tolist()
                combined_array.astype(np.float32)
                self.X_train_input.append(combined_array)

                seg_2d = seg_img[:, :, j]
                seg_2d.astype(int)
                self.X_train_target.append(seg_2d)
            del all_3d_data
            print("finished {}".format(i))

        print(" LGG Train")
        for i in self.LGG_name_train:
            all_3d_data = []
            for j in self.data_types:
                img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_data()
                img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_3d_data.append(img)

            seg_path = os.path.join(LGG_data_path, i, i + '_seg.nii.gz')
            seg_img = nib.load(seg_path).get_data()
            seg_img = np.transpose(seg_img, (1, 0, 2))
            for j in range(all_3d_data[0].shape[2]):
                combined_array = np.stack(
                    (all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]),
                    axis=2)
                combined_array = np.transpose(combined_array, (1, 0, 2))
                combined_array.astype(np.float32)
                self.X_train_input.append(combined_array)

                seg_2d = seg_img[:, :, j]
                seg_2d.astype(int)
                self.X_train_target.append(seg_2d)
            del all_3d_data
            print("finished {}".format(i))

        self.X_train_input = np.asarray(self.X_train_input, dtype=np.float32)
        self.X_train_target = np.asarray(self.X_train_target)

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 23:45:55 2021

@author: scnc
"""

import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import SimpleITK as sitk
import cv2
from dataloader.class_transformations import TransformationGeneratort
#from class_transformations import TransformationGeneratort
from skimage.transform import resize

def scalRadius(img, scale):
    x = img[int(img.shape[0]/2),:,:].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    if r < 0.001: # This is for the very black images
        r = scale*2
    s = scale*1.0/r
    return cv2.resize(img, (0,0), fx=s, fy=s)


def load_preprocess_img(dir_img):
    scale = 300
    a = cv2.imread(dir_img)
    a = scalRadius(a,scale)
    a = cv2.addWeighted(a,4,cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
    b = np.zeros(a.shape)
    cv2.circle(b, (int(a.shape[1]/2),int(a.shape[0]/2)), int(scale*0.9), (1,1,1), -1, 8, 0)
    a = a*b + 128*(1-b)
    img = Image.fromarray(np.array(a, dtype=np.int8), "RGB")
    return img

def load_dicom(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename, sitk.sitkInt16)
    # Convert the image to a  numpy array
    np_img = sitk.GetArrayFromImage(itkimage)
    np_img = np.array((np_img - np.min(np_img))/(np.max(np_img) - np.min(np_img))*255, dtype=np.int8)  # Normalize between 0 and 255
    #IMG_PX_SIZE=128##
    #np_img= resize(np_img,(IMG_PX_SIZE,IMG_PX_SIZE),anti_aliasing=True)##
    return np_img[0]


class TC(data.Dataset):
    """ Multi-Modal Dataset.
        Args:
        dir_imgs (string): Root directory of dataset where images exist.
        transform_fundus: Tranformation applied to fundus images
        is_train (bool): image for training or test
	fundus_img_size (int): Size for fundus images. i.e. 224
	sax_img_size (array): X, Y, Z dimensions for cardiac images
        ids_set (pandas class):
    """

    def __init__(self,
                dir_imgs,
                tagging_img_size,
                #sax_img_size,
                args,
                ids_set
                ):

        self.img_names = []
        self.tagging_img_size = tagging_img_size
        #self.sax_img_size = sax_img_size
        #self.path_imgs_sax = []
        #self.crop_c_min = []
        #self.crop_c_max = []
        #self.crop_r_min = []
        #self.crop_r_max = []
        self.roi_length = []
        self.crop_c_min_tagging = []
        self.crop_c_max_tagging = []
        self.crop_r_min_tagging = []
        self.crop_r_max_tagging = []
        # fundus image paths
        self.path_imgs_tagging = []
        # number fo participants
        self.num_parti = 0

        self.pad_input_tagging = TransformationGeneratort(output_size=self.tagging_img_size,
                                             output_spacing=[1, 1, 1],
                                             training=False,
                                             pixel_margin_ratio = 0.3,
                                             normalize = 0) # It could 0 or  -1
# =============================================================================
#         self.pad_input = TransformationGenerator(output_size=self.sax_img_size,
#                                              output_spacing=[1, 1, 1],
#                                              training=False,
#                                              pixel_margin_ratio = 0.3,
#                                              normalize = 0) # It could 0 or  -1
# =============================================================================


        # Obtaining repeated ED and all sax images (labels)
        for idx, ID in enumerate(ids_set.values):
            self.num_parti = self.num_parti + 1

            # Reading all fundus images per patient
            imgs_per_id = glob.glob(dir_imgs + 'tagging/' + str(int(ID[0])) + '/*.nii.gz')

            # Taking only one image orientation -> left/right
            img_21015 = [j for j in imgs_per_id if 'b2s' in j]
            #img_21015 = [j for j in imgs_per_id if '21016' in j]
            if len(img_21015) >= 1:
                imgs_per_id = img_21015[0]
                # path for fundus images
                self.path_imgs_tagging.append(imgs_per_id)
                # Image names
                self.img_names.append(imgs_per_id.split('/')[-1][:-4])
                # paths for cmr
               # self.path_imgs_sax.append(dir_imgs + 'sax/' + str(int(ID[0])) + '/' + 'image_SAX_001.vtk')
                # Coordinates
                #self.crop_c_min.append(int(ID[1]))
                #self.crop_c_max.append(int(ID[2]))
                #self.crop_r_min.append(int(ID[3]))
                #self.crop_r_max.append(int(ID[4]))
                self.roi_length.append(int(ID[5]))
                self.crop_c_min_tagging.append(int(ID[1]))
                self.crop_c_max_tagging.append(int(ID[2]))
                self.crop_r_min_tagging.append(int(ID[3]))
                self.crop_r_max_tagging.append(int(ID[4]))
            

            else:
                continue

            # # Taking max two images per participant
            # if len(imgs_per_id) > 2:
            #     list_aux = []
            #     img_21015 = [j for j in imgs_per_id if '21015' in j]
            #     list_aux.append(img_21015[-1])
            #     img_21016 = [j for j in imgs_per_id if '21016' in j]
            #     list_aux.append(img_21016[-1])
            #     imgs_per_id = list_aux

            # for n, path_fun in enumerate(imgs_per_id):
            #     # path for fundus images
            #     self.path_imgs_fundus.append(path_fun)
            #     # Image names
            #     self.img_names.append(path_fun.split('/')[-1][:-4])
            #     # paths for cmr
            #     self.path_imgs_sax.append(dir_imgs + 'sax/' + str(int(ID[0])) + '/' + 'image_SAX_001.vtk')
            #     # Coordinates
            #     self.crop_c_min.append(int(ID[1]))
            #     self.crop_c_max.append(int(ID[2]))
            #     self.crop_r_min.append(int(ID[3]))
            #     self.crop_r_max.append(int(ID[4]))
            #     self.roi_length.append(int(ID[5]))
            #     # mtd LVEDV_automatic[6], LVM_automatic[10], sex[24], dbpa[30], sbpa[31], ss[33], ads[34], bmi[36], age[38], hba1c[40], chol[41], glucose[43]
            #     self.mtdt.append([ID[6], ID[10], ID[24],  ID[30], ID[31], ID[33], ID[34], ID[36], ID[38], ID[40], ID[41], ID[43]])
            
            # Transform for fundus images
# =============================================================================
#         self.transform_fundus = transforms.Compose([
#                 transforms.Resize((self.tagging_img_size, self.tagging_img_size)),
#                 transforms.ToTensor(),
#             ])
# =============================================================================


    # Denotes the total number of samples
    def __len__(self):
        return len(self.path_imgs_tagging) # self.num_parti

    # This generates one sample of data
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (fundus, sax, label, img_name, index)
        """
        
# =============================================================================
#         # Loading fundus image
#         # preprocessing
#         tagging = load_dicom(self.path_imgs_tagging[index])
#         # without preprocessing
#         #tagging = Image.open(self.path_imgs_tagging[index]).convert('RGB')
#         # resizing the images
#         #tagging_image = self.transform_fundus(tagging)
#         # normalizing the images
#         tagging_image = tagging
#         tagging_image = (tagging_image - torch.min(tagging_image))/(torch.max(tagging_image) - torch.min(tagging_image)) # Normalize between 0 and 1
#         tagging_image = 2.0*(tagging_image - torch.min(tagging_image))/(torch.max(tagging_image) - torch.min(tagging_image))-1.0  # Normalize between -1 and 1
# =============================================================================
        tagging, output_image_np_mean,output_image_np_std, output_image_np_max = self.pad_input_tagging.get(self.path_imgs_tagging[index],
                                    self.crop_c_min_tagging[index],
                                    self.crop_c_max_tagging[index],
                                    self.crop_r_min_tagging[index],
                                    self.crop_r_max_tagging[index],
                                    self.roi_length[index])       
        
        # Loading sax
# =============================================================================
#         sax = self.pad_input.get(self.path_imgs_sax[index],
#                                     self.crop_c_min[index],
#                                     self.crop_c_max[index],
#                                     self.crop_r_min[index],
#                                     self.crop_r_max[index],
#                                     self.roi_length[index])
# =============================================================================


        #return tagging, sax,  self.img_names[index] # index  # torch.FloatTensor(self.mtdt[index])
        return tagging, self.img_names[index], output_image_np_mean,output_image_np_std, output_image_np_max # index  # torch.FloatTensor(self.mtdt[index])
        #return sax, torch.FloatTensor(self.mtdt[index]), self.img_names[index] # index  # torch.FloatTensor(self.mtdt[index])


def Tagging_loader(batch_size,
              tagging_img_size,
              #sax_img_size,
              num_workers,
              shuffle,
              dir_imgs,
              args,
              ids_set):


    ######### Create class Dataset MM ########
    TC_dataset = TC(dir_imgs = dir_imgs,
                    tagging_img_size = tagging_img_size,
		           #sax_img_size = sax_img_size,
                    args = args,
                    ids_set = ids_set)

    print('Found ' + str(len(TC_dataset)) + ' tagging images')

    # Dataloader
    data_loader = torch.utils.data.DataLoader(TC_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader

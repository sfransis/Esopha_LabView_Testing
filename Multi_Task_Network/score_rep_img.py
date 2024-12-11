import os
import torch
import pickle
import Model as net
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim.lr_scheduler
import cv2

import time
from itertools import chain
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import keyboard
import json
from datetime import datetime
import torch.nn.functional as nnf
import torch.nn as nn
import math
import numbers
import torch.nn.functional as F
import pandas as pd

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.weight = kernel.type(torch.FloatTensor)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
    
def circle_crop(imgog,mask_3, x0, x1, y0, y1, dx, dx1, dy, dy1):
    global roi_mask
    # mask_3 = roi_mask.copy() # this conversion may only be done once per roi mask 
    # mask_3[mask_3==255] = 1
    # scale = 1.60 # Conversion tablet to fast hrme)
    # mask_3 = cv2.resize(mask_3, (int(mask_3.shape[1]*scale), int(mask_3.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    #print(imgog.shape, mask_3.shape)
    print(" we also here" , imgog.shape)
    print(" and here as well", mask_3.shape)
    imgog1 = mask_3*imgog
    ## Need to find x and y min max coordinates 
    # x, y = np.where(mask_3[:,:,0]==1)
    # x0 = np.min(x)
    # x1 = np.max(x)
    # y0 = np.min(y)
    # y1 = np.max(y)
    imgog1 = imgog1[:, :, x0:x1, y0:y1]
    #background = np.zeros((1024,1024,3), dtype=np.uint8)
    background = torch.zeros(1, 3, 1024,1024).type(torch.cuda.FloatTensor)
    # dx = int(512-0.5*imgog1.shape[0])
    # dy = int(512-0.5*imgog1.shape[1])
    # if dy<0:
    #     dy=0
    # if dx<0:
    #     dx=0
    background[:, :, dx:dx1, dy:dy1] = imgog1[:, :, :1024, :1024]
    return background
    
def proccess_roi_mask(roi_mask):
    
    gpu = 0
    mask_3 = roi_mask.copy() # this conversion may only be done once per roi mask 
    mask_3[mask_3==255] = 1
    scale = 1.60 # Conversion tablet to fast hrme)
    mask_3 = cv2.resize(mask_3, (int(mask_3.shape[1]*scale), int(mask_3.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    ## Need to find x and y min max coordinates 
    x, y = np.where(mask_3[:,:,0]==1)
    x0 = np.min(x)
    x1 = np.max(x)
    y0 = np.min(y)
    y1 = np.max(y)
    imgog1 = mask_3[x0:x1, y0:y1, :]
    dx = int(512-0.5*imgog1.shape[0])
    dy = int(512-0.5*imgog1.shape[1])
    if dy<0:
        dy=0
    if dx<0:
        dx=0
    mask_3 = torch.from_numpy(mask_3)
    mask_3 = torch.moveaxis(mask_3, -1, 0)
    mask_3 = mask_3.unsqueeze(0).cuda(gpu)
    return (mask_3, x0, x1, y0, y1, dx, dx+imgog1.shape[0], dy, dy+imgog1.shape[1])

class Y_Net_Transform(object):

    gpu = 0
    
    def __init__(self, scale, param):
        self.crop = scale
        self.param = param

    def __call__(self, img):
        # Crop Image Cente
        global dst
        t = time.time()
        gpu = 0
        smoothing = GaussianSmoothing(3, 3, 3).cuda(gpu)
        scale = 1.60 # Conversion tablet to fast hrme)
        tablet_dim = 1024

        # Torch Resize
        img = torch.movedim(img, -1, 0)
        img = img.unsqueeze(0)
        

        resized = nnf.interpolate(img, scale_factor=(scale, scale), mode='bilinear', align_corners=True)
        # resized = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        resized = circle_crop(resized, self.param[0], self.param[1], self.param[2], self.param[3], self.param[4], self.param[5], self.param[6], self.param[7], self.param[8])
        #cv2.imshow('crop', resized)
        #cv2.waitKey(0)
        #print(resized)
        resized = smoothing(resized)
        #kernel = np.ones((3,3),np.float32)/9
        #resized = cv2.filter2D(resized,-1,kernel)
        edge_buffer = 200
        square_side = 312
        final_dimension = 384
        h = tablet_dim
        w = tablet_dim
        crop_img = resized[:,:, edge_buffer:h-edge_buffer, edge_buffer:w-edge_buffer]
        crop_img_1 = crop_img[:,:, 0:square_side, 0:square_side]
        crop_img_2 = crop_img[:,:,square_side:, square_side:]
        crop_img_3 = crop_img[:,:,0:square_side, square_side:]
        crop_img_4 = crop_img[:,:,square_side:, 0:square_side]
        crop_img_1 = nnf.interpolate(crop_img_1, size=(final_dimension, final_dimension), mode='bilinear', align_corners=True)
        crop_img_2 = nnf.interpolate(crop_img_2, size=(final_dimension, final_dimension), mode='bilinear', align_corners=True)
        crop_img_3 = nnf.interpolate(crop_img_3, size=(final_dimension, final_dimension), mode='bilinear', align_corners=True)
        crop_img_4 = nnf.interpolate(crop_img_4, size=(final_dimension, final_dimension), mode='bilinear', align_corners=True)
        #crop_img_1 = cv2.resize(crop_img_1, (final_dimension, final_dimension))
        #crop_img_2 = cv2.resize(crop_img_2, (final_dimension, final_dimension))
        #crop_img_3 = cv2.resize(crop_img_3, (final_dimension, final_dimension))
        #crop_img_4 = cv2.resize(crop_img_4, (final_dimension, final_dimension))
        img_list = [crop_img_1, crop_img_2, crop_img_3, crop_img_4]
        batch_imgs = torch.cat(img_list, 0)
        batch_imgs = torch.div(batch_imgs, 255)

        return batch_imgs


def process_single_image(image, json_path):

    roi_pwd = r'C:\Users\EsophaHRME_Dell2\Documents\GitHub\Esophageal_HRME_Software\EsophagealDeployment\Multi_Task_Network\HRME_Mask.png'

    roi_mask = cv2.imread(roi_pwd)
    print("Yooooo we here boibiboajfdoisa;" , roi_mask.shape)
    path_model = r'C:\Users\EsophaHRME_Dell2\Documents\GitHub\Esophageal_HRME_Software\EsophagealDeployment\Multi_Task_Network\model_30.pth'
    gpu = 0
    scaleIn = 1

    model = net.ResNetC1_YNet(2, 2, 0.1, True, 0)

    params_src = dict(torch.load(path_model, map_location='cuda:0'))
    params_dest  = model.state_dict()
    model.load_state_dict(torch.load(path_model, map_location='cuda:0'))
    dict_dest = dict(params_dest)
    for name in params_src.keys():
        if name in dict_dest.keys():
            dict_dest[name].data.copy_(params_src[name].data)

    model.load_state_dict(dict_dest)

    model = model.cuda(gpu)

    cudnn.benchmark = True
    model.eval()
    sm = torch.nn.Softmax().cuda()

    param_roi = proccess_roi_mask(roi_mask)

    valTransforms = Compose([
                Y_Net_Transform(scaleIn, param_roi)
                # ToTensor(scaleIn)
                            ])

    #global path_write, score_arr, name_arr, pass_qc_arr
    print(f"\n\nTHE IMG Passed in through labview: {image}\n\n")
    
    #start = time.time()
    with torch.no_grad():
        imagename = image
        print(f"\n\n This is the imagename: \n {imagename}\n\n")
        print(f"\n\n This is the image that has been passed in: \n {image}\n\n")
        image = cv2.imread(image)
        image = cv2.resize(image, (720, 540))
        print("\n\n\n\n YOOOOOOOOO the image size is: ", image.shape)
        print(f"\n\n This is the newly created image: \n {image}\n\n")
        # convert image directly to torch in GPU
        if image is None:
            print('Fail', imagename)
            return False 
        print(image.shape)
        image = torch.from_numpy(image).type(torch.FloatTensor).cuda(gpu)
        image_T = valTransforms(image)
        #input = image_T.cuda(gpu)
        #print('Tensor is cuda', image_T.is_cuda)
        output1 = model(image_T)
        probs = sm(output1).cpu().detach().numpy()[:, 1]
        print(probs)
        # Generate MAsk Counts
        prob_mean = np.mean(probs)
        #print(prob_mean)
        #print(str(end1) +' ' + str(end2) + ' ' +str(end3))
        # print('MeanProbs', prob_mean)
        # with open(path_write+os.sep+imagename.split(os.sep)[-1][:-4]+'.txt', 'w') as f:
        #     f.write(str(probs))
        #     f.close()
        # print(path_write+os.sep+imagename.split(os.sep)[-1][:-4]+'.json')
        with open(json_path, 'w') as jsf:
           data = {'Score': str(prob_mean), 'ImageName': imagename.split(os.sep)[-1][:-4]}
           json.dump(data, jsf)
           #print("\nTHE JSON FILE: " + path_write+os.sep+imagename.split(os.sep)[-1][:-4]+'.json\n')
        # score_arr.append(prob_mean)
        # name_arr.append(imagename.split(os.sep)[-1][:-4])
        # pass_qc_arr.append(1)
        #elapsed = time.time() - start
        return str(prob_mean)
#{"Score": "0.145597", "ImageName": "image3_14998_18"}
process_single_image(r'C:\Users\EsophaHRME_Dell2\Desktop\Training HRME\22-027 image1.png', r"C:\Users\EsophaHRME_Dell2\Desktop\Frames_Testing\runmodetest_241105\scores\image3_14998_18.json")
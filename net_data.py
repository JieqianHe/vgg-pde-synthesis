import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
from collections import OrderedDict
import caffe
base_dir = os.getcwd()
sys.path.append(base_dir+'/DeepImageSynthesis/')
from ImageSyn import ImageSyn
from Misc import *
import LossFunctions
VGGweights = os.path.join(base_dir, 'Models/vgg_normalised.caffemodel')
VGGmodel = os.path.join(base_dir, 'Models/VGG_ave_pool_deploy.prototxt')
imagenet_mean = np.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)
im_dir = os.path.join(base_dir, 'Images/')
gpu = 0
caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'
caffe.set_device(gpu)

source_img_name1 = glob.glob1(im_dir, 'pebbles.jpg')[0]
source_img_org1 = caffe.io.load_image(im_dir + source_img_name1)
im_size = 256.
[source_img1, net] = load_image(im_dir + source_img_name1, im_size, 
                            VGGmodel, VGGweights, imagenet_mean, 
                            show_img=False)

source_img_name2 = glob.glob1(im_dir, 'Brick100.jpg')[0]
source_img_org2 = caffe.io.load_image(im_dir + source_img_name2)
im_size = 256.
[source_img2, net] = load_image(im_dir + source_img_name2, im_size, 
                            VGGmodel, VGGweights, imagenet_mean, 
                            show_img=False)

im_size = np.asarray(source_img2.shape[-2:])

# In[3]:

#l-bfgs parameters optimisation
maxiter = 2000
m = 20

#define layers to include in the texture model and weights w_l
#tex_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
tex_layers = ['conv1_1', 'conv1_2', 'conv2_2', 'conv3_4','conv4_4']
tex_weights = [1e9,1e9,1e9,1e9,1e9]

#pass image through the network and save the constraints on each layer
constraints1 = OrderedDict()
net.forward(data = source_img1)
for l,layer in enumerate(tex_layers):
    #constraints1[layer] = constraint([LoslsFunctions.gram_mse_loss],
     #                               [{'target_gram_matrix': gram_matrix(net.blobs[layer].data),
      #                               'weight': tex_weights[l]}])
    np.save('net_data1_' + layer + '.npy', net.blobs[layer].data)

constraints2 = OrderedDict()
net.forward(data = source_img2)
for l,layer in enumerate(tex_layers):
    np.save('net_data2_' + layer + '.npy', net.blobs[layer].data)

filter_layers = ['conv1_1', 'conv1_2', 'conv2_2', 'conv3_4', 'conv4_4']
for l, layer in enumerate(filter_layers):
    np.save('net_filter_' + layer + '.npy', net.params[layer][0].data)

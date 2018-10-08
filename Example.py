
# coding: utf-8

# In[1]:

#get_ipython().magic('pylab inline')
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
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


# In[2]:

#load source image
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

source_img_name3 = glob.glob1(im_dir, 'NewInit.jpg')[0]
source_img_org3 = caffe.io.load_image(im_dir + source_img_name1)
im_size = 256.
[source_img3, net] = load_image(im_dir + source_img_name1, im_size, 
                            VGGmodel, VGGweights, imagenet_mean, 
                            show_img=False)
                              
im_size = np.asarray(source_img2.shape[-2:])

# In[3]:

#l-bfgs parameters optimisation
maxiter = 2000
m = 20

#define layers to include in the texture model and weights w_l
tex_layers = ['conv4_4', 'conv3_4', 'conv2_2', 'conv1_2', 'conv1_1']
tex_weights = [1e9,1e9,1e9,1e9,1e9]
#tex_layers = ['conv2_2', 'conv1_2', 'conv1_1']
#tex_weights = [1e9,1e9,1e9]
diag = True

for i in range(11):
    alpha = i
    #pass image through the network and save the constraints on each layer
    constraints1 = OrderedDict()
    net.forward(data = source_img1)
    for l,layer in enumerate(tex_layers):
        constraints1[layer] = constraint([LossFunctions.gram_mse_loss],
                                        [{'target_gram_matrix': gram_matrix(net.blobs[layer].data),
                                         'diag': diag,
                                         'weight': tex_weights[l]}])

    constraints2 = OrderedDict()
    net.forward(data = source_img2)
    for l,layer in enumerate(tex_layers):
        constraints2[layer] = constraint([LossFunctions.gram_mse_loss],
                                        [{'target_gram_matrix': gram_matrix(net.blobs[layer].data),
                                         'diag': diag,
                                         'weight': tex_weights[l]}])
    #get optimisation bounds
    bounds = get_bounds([source_img1],im_size)

    #generate new texture
    result = ImageSyn(net, constraints1, constraints2, alpha = alpha, bounds=bounds, 
                      callback=lambda x: show_progress(x,net), 
                      minimize_options={'maxiter': maxiter,
                                        'maxcor': m,
                                        'ftol': 0, 'gtol': 0})


    # In[4]:

    #match histogram of new texture with that of the source texture and show both images
    new_texture = result['x'].reshape(*source_img1.shape[1:]).transpose(1,2,0)[:,:,::-1]
    new_texture = histogram_matching(new_texture, source_img_org1)
    plt.imshow(new_texture)
    plt.savefig('new_5layer_diag_init%s.png'%i)
    



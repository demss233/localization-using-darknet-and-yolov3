import sys
import os

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
from torch.autograd import Variable

import cv2
import matplotlib.pyplot as plt

from src.darknet import Darknet 
from src.utils import load_classes
from src.utils import prep_image
from src.non_max_suppression import write_results

batch_size = 2

weightsfile = './weights/yolov3.weights'
classfile = './config/coco.names'
cfgfile = './config/yolov3.cfg'
input_dir = './input'
output_dir = './test'
nms_thesh = 0.5
CUDA = False

if not os.path.exists(output_dir):
    print("Output directory not found! \nCreating the /test directory...")
    os.mkdir(output_dir)
    print("/test directory successfully created")

print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print("Network successfully loaded")

classes = load_classes(classfile)
print('Classes loaded')

inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

try:
    imlist = [os.path.join(os.path.realpath('.'), input_dir, img) for img in os.listdir(input_dir)]
except NotADirectoryError:
    imlist = [os.path.join(os.path.realpath('.'), input_dir)]
except FileNotFoundError:
    print("No file or directory with the name {}".format(input_dir))
    exit()

batches = list(map(prep_image, imlist, [inp_dim] * len(imlist)))
im_batches = [x[0] for x in batches] 
orig_ims = [x[1] for x in batches] 
im_dim_list = [x[2] for x in batches]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2) 

if CUDA:
    im_dim_list = im_dim_list.cuda()

reminder = 1 if len(im_dim_list) % batch_size else 0
num_batches = len(imlist) // batch_size + reminder
im_batches = [torch.cat(im_batches[i * batch_size:min((i + 1) * batch_size, len(im_batches))]) 
              for i in range(num_batches)]

output_all = []
for i, batch in enumerate(im_batches):
    if CUDA:
        batch = batch.cuda()       
    with torch.no_grad():
        preds = model(Variable(batch), CUDA)  # renamed from `prediction` to avoid conflicts
    preds = write_results(preds, confidence = 0.5, num_classes = 80, nms_conf = nms_thesh)
    if type(preds) == int:
        continue
    preds[:, 0] += i * batch_size  
    output_all.append(preds)

if not output_all:
    print("No detections were made")
    exit()

output = torch.cat(output_all)

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:,0].view(-1,1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:,1].view(-1,1)) / 2
output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

def write_box(x, results):
    c1 = (int(x[1].item()), int(x[2].item()))
    c2 = (int(x[3].item()), int(x[4].item()))
    img = results[int(x[0].item())]
    cls = int(x[-1].item())
    label = classes[cls]
    color = (0,0,255)

    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2_text = (c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4)
    cv2.rectangle(img, c1, c2_text, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, (225,255,255), 1)
    return img

list(map(lambda x: write_box(x, orig_ims), output))
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(output_dir, os.path.basename(x)))
list(map(cv2.imwrite, det_names, orig_ims))
torch.cuda.empty_cache()

# Example
img = cv2.imread('./input/office.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (20,10))
plt.imshow(img)
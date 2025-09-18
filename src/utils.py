import torch
import torch.nn as nn
import numpy as np
import cv2

def parse_config(fpath):
    file = open(fpath, 'r')
    file = file.read().split('\n')
    file = [line for line in file if len(line) > 0 and line[0] != '#']
    file = [line.lstrip().rstrip() for line in file]

    blocks_list = []
    current_dict = {}
    for line in file:
        if line[0] == '[':
            if len(current_dict) != 0:
                blocks_list.append(current_dict)
                current_dict = {}
            current_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])
        else:
            temp = line.split('=')
            current_dict[temp[0].rstrip()] = temp[1].lstrip()
    blocks_list.append(current_dict)
    return blocks_list

def prediction_helper(x, input_dim, anchors, num_classes, CUDA = False):
    batch_size = x.size(0)
    grid_size = x.size(2)
    stride = input_dim // grid_size
    bbox_attribs = 5 + num_classes
    num_anchors = len(anchors)

    pred = x.view(batch_size, bbox_attribs * num_anchors, grid_size * grid_size)
    pred = pred.transpose(1, 2).contiguous()
    pred = pred.view(batch_size, grid_size * grid_size * num_anchors, bbox_attribs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    pred[:, :, 0] = torch.sigmoid(pred[:, :, 0])
    pred[:, :, 1] = torch.sigmoid(pred[:, :, 1])
    pred[:, :, 4] = torch.sigmoid(pred[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    pred[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    pred[:, :, 2 : 4] = torch.exp(pred[:, :, 2 : 4]) * anchors
    pred[:, :, 5 : 5 + num_classes] = torch.sigmoid((pred[:, :, 5 : 5 + num_classes]))
    pred[:, :, :4] *= stride

    return pred

def intersection_over_union(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min = 0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min = 0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def prep_image(img, inp_dim):
    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2 : (h - new_h) // 2 + new_h, (w - new_w) // 2 : (w - new_w) // 2 + new_w, :] = resized_image
    
    return canvas
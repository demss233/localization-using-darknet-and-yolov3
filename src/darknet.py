import torch
import torch.nn as nn
import numpy as np
from .utils import parse_config, prediction_helper
from .non_max_suppression import write_results

class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()
        
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def formulate_model(blocks):
    darknet_details = blocks[0]
    channels = 3
    output_filters = []
    modules = nn.ModuleList()

    for i, block in enumerate(blocks[1:]):
        net = nn.Sequential()
        if (block["type"] == "convolutional"):
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            use_bias = False if ("batch_normalize" in block) else True
            pad = (kernel_size - 1) // 2

            conv = nn.Conv2d(
                in_channels = channels, 
                out_channels = filters, 
                kernel_size = kernel_size,
                stride = strides, 
                padding = pad, 
                bias = use_bias
            )

            net.add_module("conv_{0}".format(i), conv)
            if "batch_normalize" in block:
                bn = nn.BatchNorm2d(filters)
                net.add_module("batch_norm_{0}".format(i), bn)

            if activation == "leaky":
                leaky = nn.LeakyReLU(0.1, inplace = True)
                net.add_module("leaky_{0}".format(i), leaky)
        
        elif (block["type"] == "upsample"):
            upconv = nn.Upsample(scale_factor = 2, mode = "bilinear")
            net.add_module("upsample_{}".format(i), upconv)

        elif (block["type"] == "route"):
            block['layers'] = block['layers'].split(',')
            block['layers'][0] = int(block['layers'][0])
            start = block['layers'][0]
            
            if len(block['layers']) == 1:
                filters = output_filters[i + start]

            elif len(block['layers']) > 1:
                block['layers'][1] = int(block['layers'][1]) - i
                end = block['layers'][1]
                filters = output_filters[i + start] + output_filters[i + end]

            route = DummyLayer()
            net.add_module("route_{0}".format(i), route)

        elif block["type"] == "yolo":
            mask = block['mask'].split(",")
            mask = [int(m) for m in mask]
            anchors = block["anchors"].split(',')
            anchors = [(int(anchors[i]), (int(anchors[i + 1]))) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            block["anchors"] = anchors
            detector = DetectionLayer(anchors)
            net.add_module("Detection_{0}".format(i), detector)

        modules.append(net)
        output_filters.append(filters)
        channels = filters

    return darknet_details, modules

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_config(cfgfile)
        self.net_info, self.module_list = formulate_model(self.blocks)

    def forward(self, x, CUDA=False):
        modules_ = self.blocks[1:]
        outputs = {}
        write = 0

        for i, module in enumerate(modules_):
            module_type = module["type"]

            if module_type in ["convolutional", "upsample"]:
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = [int(a) for a in module["layers"]]
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = prediction_helper(x.data, input_dim, anchors, num_classes)

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]

        return detections if write else 0

    def load_weights(self, weightfile):
        file = open(weightfile, 'rb')
        header = np.fromfile(file, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(file, dtype=np.float32)
        ptr = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]

                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]); ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]); ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]); ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]); ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases]); ptr += num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights]); ptr += num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
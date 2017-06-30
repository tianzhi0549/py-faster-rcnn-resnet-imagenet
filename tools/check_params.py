import _init_paths
import caffe
import argparse
import numpy as np
import os, sys

prototxt = "models/imagenet/ResNet-101/faster_rcnn_alt_opt/rpn_test.pt"

# weights1 = "data/imagenet_models/ResNet-101.caffemodel"
weights1 = "/media/sdb/zhitian/code/py-faster-rcnn-resnet/output/faster_rcnn_alt_opt/imagenet_2015_trainval1_woextra/resnet-101_fast_rcnn_stage2_iter_320000.caffemodel"
weights2 = "/media/sdb/zhitian/code/py-faster-rcnn-resnet/output/faster_rcnn_alt_opt/imagenet_2015_trainval1_woextra/models_bp4/resnet-101_rpn_stage2_iter_320000.caffemodel"

net1 = caffe.Net(prototxt, weights1, caffe.TEST)
net2 = caffe.Net(prototxt, weights2, caffe.TEST)

for layer_name in net1.params:
	params1 = net1.params[layer_name]
	params2 = net2.params[layer_name]
	for index in range(len(params1)):
		if np.all(params1[index].data == params2[index].data):
			pass	
		else:
			print layer_name, "No", np.sum(np.abs(params1[index].data - params2[index].data))

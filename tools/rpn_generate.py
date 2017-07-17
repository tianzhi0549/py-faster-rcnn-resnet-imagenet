#!/usr/bin/env python

# --------------------------------------------------------
# Fast/er/ R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Generate RPN proposals."""

import _init_paths
import numpy as np
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import cPickle
import caffe
import argparse
import pprint
import time, os, sys
import multiprocessing as mp
from easydict import EasyDict

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu',  type=int, nargs='+', 
                        default=[0], help="List of device ids.")
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def rpn_generate_single_gpu(prototxt, caffemodel, imdb, rank, gpus, output_dir):
    cfg.GPU_ID = gpus[rank]
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    imdb_boxes = imdb_proposals(net, imdb, rank, len(gpus), output_dir)

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # RPN test settings
    cfg.TEST.RPN_PRE_NMS_TOP_N = -1
    cfg.TEST.RPN_POST_NMS_TOP_N = 300

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    fake_net = EasyDict()
    fake_net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    gpus = args.gpu

    imdb = get_imdb(args.imdb_name)
    output_dir = get_output_dir(imdb, fake_net)
    output_dir = os.path.join(output_dir, "proposals_test")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    procs=[]
    for rank in range(len(gpus)):
        p = mp.Process(target=rpn_generate_single_gpu,
                    args=(args.prototxt, args.caffemodel, imdb, rank, gpus, output_dir))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

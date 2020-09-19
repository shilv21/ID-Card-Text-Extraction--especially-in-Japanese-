from collections import OrderedDict
import sys
import os
sys.path.append(os.path.dirname(__file__))

from craft import CRAFT
import zipfile
import json
import file_utils
import imgproc
import craft_utils
import numpy as np
from skimage import io
import cv2
from django.conf import settings
from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import time
import argparse


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def getOriginImage(imagePath):
    return cv2.imread(imagePath)

def is_SubList (Container, subList):
    for ele in subList:
        if ele not in Container:
            return False
    return True

def is_Box_In_Lines (box, lines):
    for line in lines:
        if box in line:
            return False
    return True

def getFirstPoint(val):
    return val[0]

def runLineCut(imagePath, net, resultsPath):
    image = imgproc.loadImage(imagePath)
    maxsize = 500
    if image.shape[0] > maxsize:
        image = cv2.resize(
            image, (maxsize, int(image.shape[1]*maxsize/image.shape[0])))
    if image.shape[1] > maxsize:
        image = cv2.resize(
            image, (int(image.shape[0]*maxsize/image.shape[1]), maxsize))
    oririnImage = getOriginImage(imagePath)
    bboxes, polys, score_text = test_net(
        net, image, 0.7, 0.4, 0.4, True, False)
    lines=[]
    while True:
        is_exit = False
        boxInLineTotal = []
        for y in range(image.shape[1]):
            boxInLine = []
            for i, box in enumerate(polys):
                box = np.array(box).astype(np.int32).tolist()
                if is_Box_In_Lines(box,lines):
                    firstPoint = box[0]
                    endPoint = box[2]
                    if y>=firstPoint[1] and y<=endPoint[1]:
                        boxInLine.append(box)
            if is_SubList(boxInLine,boxInLineTotal):
                boxInLineTotal =  boxInLine     
        if boxInLineTotal not in lines:
            is_exit = True
            lines.append(boxInLineTotal)
        if not is_exit:
            break
    for i,line in enumerate(lines):
        line = sorted(line,key = lambda box: box[0])
        for ii, box in enumerate(line):
            firstPoint = box[0]
            endPoint = box[2]
            imageCroped = image[firstPoint[1]:endPoint[1], firstPoint[0]:endPoint[0]]
            cv2.imwrite(resultsPath+'/' + str(i)+'_' + str(ii) + '.png', imageCroped)
    return bboxes, polys, score_text


def createModel():
    net = CRAFT()

    weightPath = os.path.join(
        settings.BASE_DIR, 'CRAFT/weights/craft_mlt_25k.pth')

    print('Loading weights from checkpoint (' + weightPath + ')')

    net.load_state_dict(copyStateDict(torch.load(weightPath)))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    net.eval()
    return net

    # bboxes, polys, score_text = runLineCut('./data/test.png', net, './result')

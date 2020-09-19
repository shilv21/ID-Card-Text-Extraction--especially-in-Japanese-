import sys
import os
sys.path.append(os.path.dirname(__file__))
from django.conf import settings
import time
from ocr_baki.OCR_model import AlpsOcrModel
import unicodedata
import glob
import re
import io
import pandas as pd
import numpy as np
from PIL import Image
import copy



base_folder = os.path.dirname(os.path.abspath(__file__))


def getReadTextModel():
    ocr_engine_v0 = AlpsOcrModel()
    ocr_engine_v0.load(weights_path=os.path.join(base_folder, 'ocr_baki/baki_tcn_model.hdf5'),
                       label_text_path=os.path.join(base_folder, 'ocr_baki/baki_tcn_label.json'))
    return ocr_engine_v0


def _pre_process(img, hsize=48):
    img = img.convert('L')
    w, h = img.size
    img = img.resize((int(hsize / h * w), hsize), Image.ANTIALIAS)
    w, h = img.size
    img_4d = np.array(img).reshape(-1, h, w, 1)
    return img_4d / 255

    pass


def runReadText(inputPath, ocr_engines):
    f = []
    result = ''
    listFile = [f for f in os.listdir(
        inputPath) if os.path.isfile(os.path.join(inputPath, f))]
    listFileRowCol = []
    for fileName in listFile:
        fileas= []
        fileName = fileName.replace('.png','')
        row_Col = fileName.split('_')
        for i in range(len(row_Col)):
            row_Col[i] =  int(row_Col[i])
            fileas.append(row_Col[i])
        fileName += '.png'
        fileName = os.path.join(inputPath, fileName)
        fileas.append(fileName)
        listFileRowCol.append(fileas)
    
    listFileRowCol = sorted(listFileRowCol, key = lambda fileInfo: [fileInfo[0],fileInfo[1]])
    for i in range(len(listFile)):
        listFile[i] =  os.path.join(inputPath,listFile[i])
    oldrow = 0
    for i, sth in enumerate(listFileRowCol):
        if sth[0] != oldrow:
            oldrow  = copy.deepcopy(sth[0])
            result+='<br>'
        
        
        image = _pre_process(Image.open(sth[2]))
        image = [image]
        ocr_result = ocr_engines.predict_batch(image)
        for text in ocr_result:
            for t in text:
                result += t+' '
        
    # for i, filePath in enumerate(listFile):
    #     image = _pre_process(Image.open(filePath))
    #     image = [image]
    #     ocr_result = ocr_engines.predict_batch(image)
    #     for text in ocr_result:
    #         for t in text:
    #             result += t+' '
    return result

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:27:27 2019

@author: samas
"""

import os
import re
from PIL import Image
from pdf2image import convert_from_path



path = "/Users/raghuram.b/Desktop/single_page_A4Size_jpegs/"
path1 = "/Users/raghuram.b/Desktop/A4_SINGLEPAGE/"
Image.MAX_IMAGE_PIXELS = 9999999999

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
#Image.errors.simplefilter('error', PDFPageCountError)


# A4 size 2480 x 3508

fname = []
for root,d_names,f_names in os.walk(path):
    for f in f_names:
        fname.append(os.path.join(root, f))
#print("fname = %s" %fname)
for fnam in fname :
    print(fnam)
    
    img = Image.open(fnam) # image extension *.png,*.jpg
    new_width  = 2480
    new_height = 3508
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    fnam=fnam.split('/')[-1]
    img.save(path1+fnam) # format may what u want ,*.png,*jpg,*.gif

 
#              
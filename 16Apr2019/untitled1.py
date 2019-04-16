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



path = "/Users/raghuram.b/Desktop/data_new/validation/PMOC/"
path1 = "/Users/raghuram.b/Desktop/data_new/validation/PMOC/"
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
    if fnam.endswith('.pdf') or fnam.endswith('.PDF') :
        jpgnam = re.sub('\.pdf$', '.jpg', fnam,flags= re.IGNORECASE)
        pages = convert_from_path(fnam, 500) ########################################
        new_im = Image.new('RGB', (2480,3508),'white')
#        if(len(pages)==1):
        img=pages[0].resize((2480,3508))
        new_im.paste(img,(0,0,2480,3508)) 
        new_im.save(path1+jpgnam.split('/')[-1],'JPEG')
#        elif(len(pages)==2):
#           img= pages[0].resize((2480,3508))
#           img1=pages[1].resize((2480,3508))           
#           new_im.paste(img,(0,0,2480,3508))
#           new_im.paste(img1,(0,3508,2480,7016))               
#           new_im.save(path1+jpgnam.split('/')[-1],'JPEG')
#        else:
#            img=pages[0].resize((2480,3508))
#            img1=pages[1].resize((2480,3508))
#            img2=pages[2].resize((2480,3508))         
#            new_im.paste(img,(0,0,2480,3508))
#            new_im.paste(img1,(0,3508,2480,7016))  
#            new_im.paste(img2,(0,7016,2480,10524))             
#            new_im.save(path1+jpgnam.split('/')[-1],'JPEG')                
#            
        
path = "/Users/raghuram.b/Desktop/data_new/validation/RCRI/"
path1 = "/Users/raghuram.b/Desktop/data_new/validation/RCRI/"
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
    if fnam.endswith('.pdf') or fnam.endswith('.PDF') :
        jpgnam = re.sub('\.pdf$', '.jpg', fnam,flags= re.IGNORECASE)
        pages = convert_from_path(fnam, 500) ########################################
        new_im = Image.new('RGB', (2480,3508),'white')
#        if(len(pages)==1):
        img=pages[0].resize((2480,3508))
        new_im.paste(img,(0,0,2480,3508)) 
        new_im.save(path1+jpgnam.split('/')[-1],'JPEG')

              
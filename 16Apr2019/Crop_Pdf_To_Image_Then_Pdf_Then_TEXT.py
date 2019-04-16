
#PDF to DIRECT TEXT
import PyPDF2
from PyPDF2 import PdfFileWriter,PdfFileReader,PdfFileMerger

#pdf_file = PdfFileReader(open("/Users/raghuram.b/Desktop/BIRF-EM-DCCAL-510-0003.PDF","rb"))
#page = pdf_file.getPage(0)
#print(page.cropBox.getLowerLeft())
#print(page.cropBox.getLowerRight())
#print(page.cropBox.getUpperLeft())
#print(page.cropBox.getUpperRight())
#
#page.mediaBox.lowerRight = (lower_right_new_x_coordinate, lower_right_new_y_coordinate)
#page.mediaBox.lowerLeft = (lower_left_new_x_coordinate, lower_left_new_y_coordinate)
#page.mediaBox.upperRight = (upper_right_new_x_coordinate, upper_right_new_y_coordinate)
#page.mediaBox.upperLeft = (upper_left_new_x_coordinate, upper_left_new_y_coordinate)


with open("/Users/raghuram.b/Desktop/junk/pdf/BIRF-ZZ-MDZZZ-510-0007.pdf","rb") as in_f:
    input1 = PdfFileReader(in_f)
    output = PdfFileWriter()
    
    numPages = input1.getNumPages()
    print ("document has %s pages.",numPages)
    
    numPages=1
    for i in range(numPages):
        page = input1.getPage(i)
        print(page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y())
        print(page.mediaBox.getLowerRight_x(), page.mediaBox.getLowerLeft_y())
        page.trimBox.upperRight = (1350, 786)
        page.trimBox.lowerLeft = (0, 600)
        page.cropBox.upperRight = (1350, 786)
        page.cropBox.lowerLeft = (0, 600)
#        print(page.cropBox)
        text=page.extractText()
        print(text)
        output.addPage(page)
        
    with open("/Users/raghuram.b/Desktop/junk/pdf_out/BIRF-ZZ-MDZZZ-510-0007.pdf", "wb") as out_f:
        output.write(out_f)

###############
#PDF to IMAGE

import os
import re
from PIL import Image
from pdf2image import convert_from_path

path='/Users/raghuram.b/Desktop/junk/pdf_out/'
path1 = "/Users/raghuram.b/Desktop/junk/self_img_txt/"
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
        jpgnam = re.sub('\.pdf$', '.png', fnam,flags= re.IGNORECASE)
        pages = convert_from_path(fnam, 500) ########################################
#        new_im = Image.new('RGB', (2480,3508),'white')
        if(len(pages)==1):
            img=pages[0].resize((2480,2000))
            new_im.paste(img,(0,0,2480,2000)) 
            new_im.save(path1+jpgnam.split('/')[-1],quality=5000)
#    if fnam.endswith('.pdf') or fnam.endswith('.PDF') :
#        jpgnam = re.sub('\.pdf$', '.jpg', fnam,flags= re.IGNORECASE)
#        pages = convert_from_path(fnam, 5000) ########################################
#        new_im = Image.new('RGB', (2480,10524),'white')
#        if(len(pages)==1):
#            img=pages[0].resize((2480,3508))
#            new_im.paste(img,(0,0,2480,3508)) 
#            new_im.save(path1+jpgnam.split('/')[-1],'JPEG')
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


#############
#CROP IMAGE
##############
from PIL import Image
def crop(image_path, coords, saved_location):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location,1200)
    cropped_image.show()
if __name__ == '__main__':
    image = '/Users/raghuram.b/Desktop/junk/self_img_txt/BIRF-ZZ-MDZZZ-510-0007.png'
    crop(image, (161, 166, 706, 1050), '/Users/raghuram.b/Desktop/junk/self_img_txt/cropped_BIRF-ZZ-MDZZZ-510-0007.png')
    

from PIL import Image
import pyresizeimage
from pyresizeimage import resizeimage
image_path = '/Users/raghuram.b/Desktop/junk/self_img_txt/BIRF-ZZ-MDZZZ-510-0007.png'
coords=(161, 166, 706, 1050)
image_obj = Image.open(image_path)
cropped_image = image_obj.crop(coords)
cropped_image = resizeimage.resize_cover(cropped_image, [2580, 3580])
cropped_image.save('/Users/raghuram.b/Desktop/junk/self_img_txt/cropped_BIRF-ZZ-MDZZZ-510-0007.png', format)

cropped_image.show()
  
#############
#IMAGE TO PDF
##############
import img2pdf 
from PIL import Image 
import os 
  
# storing image path 
img_path = "/Users/raghuram.b/Desktop/junk/self_img_txt/cropped_BIRF-ZZ-MDZZZ-510-0007.png"
  
# storing pdf path 
pdf_path = "/Users/raghuram.b/Desktop/junk/pdf_out/cropped_BIRF-ZZ-MDZZZ-510-0007.pdf"
  
# opening image 
image = Image.open(img_path) 
  
# converting into chunks using img2pdf 
pdf_bytes = img2pdf.convert(image.filename, dpi=1200)   


# opening or creating pdf file 
file = open(pdf_path, "wb") 
  
# writing pdf files with chunks 
file.write(pdf_bytes) 
  
# closing image file 
image.close() 
  
# closing pdf file 
file.close() 
  
# output 
print("Successfully made pdf file")             
            
       
###########################
#CROPPED PDF to DIRECT TEXT
###########################
import PyPDF2
from PyPDF2 import PdfFileWriter,PdfFileReader,PdfFileMerger


with open("/Users/raghuram.b/Desktop/junk/pdf_out/cropped_BIRF-ZZ-MDZZZ-510-0007.pdf","rb") as in_f:
    input1 = PdfFileReader(in_f)
    output = PdfFileWriter()
    
    numPages = input1.getNumPages()
    print ("document has %s pages.",numPages)
    
    numPages=1
    for i in range(numPages):
        page = input1.getPage(i)
#        print(page.cropBox)
        text=page.extractText()
        print(text)

#import glob2
#import PyPDF2
#
#ocr_path='/Users/raghuram.b/Desktop/junk/'
#
#filelist=glob2.glob("/Users/raghuram.b/Desktop/out.pdf")
#for path in filelist:
#    content=''
#    pdf=PyPDF2.PdfFileReader(path,'rb')
#    pages=pdf.getNumPages()
#    for i in range(0,pages):    
#        content += pdf.getPage(i).extractText() + "\n"
#        ext=path.split('/')[-1] 
#        ext=ext+"2textnow"+".txt" 
#        ext=ext.replace("pdf","txt") 
#        ext=ext.replace("PDF","txt")
#        with open(ocr_path+ext,mode='w') as file:
#            file.write(content+"\n")
#  
#
#
#
## import the Python Image processing Library
#import glob2
#from PIL import Image
#import pytesseract
##from pytesseract import image_to_string
#Image.MAX_IMAGE_PIXELS = 978956970
#Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
##
##pytesseract.pytesseract.tesseract_cmd = r'/anaconda3/lib/python3.6/site-packages/pytesseract-0.1.7-py3.6.egg'
#
#path='/Users/raghuram.b/Desktop/junk/jpeg/*'
#
#
#filelist=glob2.glob(path)
#for FILENAME in filelist:
#   print(FILENAME)
#   # Create an Image object from an Image
#   image = Image.open(FILENAME)
#   text=pytesseract.image_to_string(image)
#   print(text)
#   ext=FILENAME.split('/')[-1]
#   ext=ext.replace("jpg","txt") 
#   with open(path+ext,mode='w') as file:
#       content=str(text)
#       file.write(content+"\n")
#
#    



   
from PyPDF2 import PdfFileWriter,PdfFileReader,PdfFileMerger

#pdf_file = PdfFileReader(open("/Users/raghuram.b/Desktop/BIRF-EM-DCCAL-510-0003.PDF","rb"))
#page = pdf_file.getPage(0)
#print(page.cropBox.getLowerLeft())
#print(page.cropBox.getLowerRight())
#print(page.cropBox.getUpperLeft())
#print(page.cropBox.getUpperRight())
#
#page.mediaBox.lowerRight = (lower_right_new_x_coordinate, lower_right_new_y_coordinate)
#page.mediaBox.lowerLeft = (lower_left_new_x_coordinate, lower_left_new_y_coordinate)
#page.mediaBox.upperRight = (upper_right_new_x_coordinate, upper_right_new_y_coordinate)
#page.mediaBox.upperLeft = (upper_left_new_x_coordinate, upper_left_new_y_coordinate)
#

with open('/Users/raghuram.b/Desktop/junk/pdf/BIRF-ZZ-MDZZZ-510-0007.pdf',"rb") as in_f:
    input1 = PdfFileReader(in_f)
    output = PdfFileWriter()
    
    numPages = input1.getNumPages()
    print ("document has %s pages.",numPages)
    
    numPages=1
    for i in range(numPages):
        page = input1.getPage(i)
        print(page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y())
        page.trimBox.lowerLeft = (0, 500)
        page.trimBox.upperRight = (225, 500)
        page.cropBox.lowerLeft = (0, 150)
        page.cropBox.upperRight = (200, 200)
        output.addPage(page)
        
    with open("/Users/raghuram.b/Desktop/out.pdf", "wb") as out_f:
        output.write(out_f)
# import the Python Image processing Library

from PIL import Image

 

# Create an Image object from an Image

imageObject  = Image.open("/Users/raghuram.b/Desktop/BIRF-AH-QRAUD-100-1003.jpg")

 

# Crop the iceberg portion

cropped     = imageObject.crop((0, 0, 499, 75))

 

# Display the cropped portion

cropped.show()

cropped.save("/Users/raghuram.b/Desktop/raghu.jpg")


import PyPDF2
import glob2

ocr_path='/Users/raghuram.b/Desktop/raghu.jpg'
pdf=PyPDF2.PdfFileReader(path,'rb')
pages=pdf.getNumPages()
for i in range(0,pages):    
    content += pdf.getPage(i).extractText() + "\n"
    ext=path.split('/')[-1] 
    ext=ext.replace("pdf","txt") 
    ext=ext.replace("PDF","txt")
    with open(ocr_path+ext,mode='w') as file:
        file.write(content+"\n")


#filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/LATEST(Jan 15)/PDFs/*")
for path in filelist:
    content=''
    pdf=PyPDF2.PdfFileReader(path,'rb')
    pages=pdf.getNumPages()
    for i in range(0,pages):    
        content += pdf.getPage(i).extractText() + "\n"
        ext=path.split('/')[-1] 
        ext=ext.replace("pdf","txt") 
        ext=ext.replace("PDF","txt")
        with open(ocr_path+ext,mode='w') as file:
            file.write(content+"\n")
  

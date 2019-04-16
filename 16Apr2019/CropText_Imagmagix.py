import glob2
from PIL import Image
import pytesseract


ocr_path='/Users/raghuram.b/Desktop/Exxon/IntelligentText/TextClassification_Crop_Only_Firstpage/TXTs/'
single_page_path='/Users/raghuram.b/Desktop/Exxon/IntelligentText/TextClassification_Crop_Only_Firstpage/single_page_A4Size_jpegs/*.jpg'


filelist=glob2.glob(single_page_path)
for FILENAME in filelist:
    print(FILENAME)
    text = pytesseract.image_to_string(Image.open(FILENAME))
    print(text)
    content=str(text)
    ext=FILENAME.split('/')[-1]
    ext=ext.replace("jpg","txt") 
    with open(ocr_path+ext,mode='w') as file:
            file.write(content)
    
    
#filelist=glob2.glob(words_path)
#for FILENAME in filelist:
#    print(FILENAME)
#    text = pytesseract.image_to_string(Image.open(FILENAME))
#    print(text)
#    content=str(text)
#    ext=FILENAME.split('/')[-1]
#    ext=ext.replace("jpg","txt") 
#    with open(ocr_path+ext,mode='w') as file:
#            file.write(content)    
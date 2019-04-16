# import the Python Image processing Library
import glob2
from PIL import Image
from PIL import Image
import PyPDF2
import pytesseract
from PyPDF2 import PdfFileWriter,PdfFileReader,PdfFileMerger
#path='/Users/raghuram.b/Desktop/BIRF-EM-BPMOC-510-0013.jpg'


filelist=glob2.glob("/Users/raghuram.b/Desktop/samplepdf/PDF/*.jpg")
for FILENAME in filelist:
    print(FILENAME)
    text = pytesseract.image_to_string(Image.open(FILENAME))
    print(text)

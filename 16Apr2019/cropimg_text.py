# import the Python Image processing Library
import glob2
from PIL import Image
from PIL import Image
from pytesseract import image_to_string
Image.MAX_IMAGE_PIXELS = 978956970
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

#path='/Users/raghuram.b/Desktop/BIRF-EM-BPMOC-510-0013.jpg'


filelist=glob2.glob("/Users/raghuram.b/Desktop/junk/pdf/*")
#FILENAME="/Users/raghuram.b/Desktop/Exxon/self_ocr_crop/test/BIRF-EM-BPMOC-510-0013.jpg"
for FILENAME in filelist:
    print(FILENAME)
    # Create an Image object from an Image
    imageObject  = Image.open(FILENAME)
       
    # Crop the iceberg portion
    
    cropped     = imageObject.crop((0, 0, 2480, 600))
    # Display the cropped portion
#    cropped.show()
    cropped.save("/Users/raghuram.b/Desktop/raghu.jpg")
    image = Image.open('/Users/raghuram.b/Desktop/raghu.jpg', mode='r')
    page1_text=image_to_string(image)
    
    cropped     = imageObject.crop((0, 3590, 2480, 4000))
    # Display the cropped portion
#    cropped.show()
    cropped.save("/Users/raghuram.b/Desktop/junk/self_img_txt/raghu.jpg")
    image = Image.open('/Users/raghuram.b/Desktop/raghu.jpg', mode='r')
#    page2_text=image_to_string(image)
#    text=page1_text+page2_text
#    print(text)

    
  
  
     
    
    







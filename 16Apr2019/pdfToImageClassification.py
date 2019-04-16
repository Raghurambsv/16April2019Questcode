import os
import re
from PIL import Image
from pdf2image import convert_from_path

path = "/Users/raghuram.b/Desktop/Exxon/Exxon_PDF(Single Page)/PDFs/"
#path = "/Users/raghuram.b/Desktop/Exxon/Exxon_PDF/Images/"

Image.MAX_IMAGE_PIXELS = 978956970

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

fname = []
for root,d_names,f_names in os.walk(path):
	for f in f_names:
		fname.append(os.path.join(root, f))

for fnam in fname :
    print(fnam)
    if fnam.endswith('.pdf') or fnam.endswith('.PDF') :
        pages = convert_from_path(fnam, 500)
        jpgnam = re.sub('\.pdf$', '.jpg', fnam,flags=re.IGNORECASE)
        jpgnam=jpgnam.split('/')[-1]
        pages[0].save('/Users/raghuram.b/Desktop/Exxon/Exxon_PDF(Single Page)/Images(Single Page)/'+jpgnam, 'JPEG')







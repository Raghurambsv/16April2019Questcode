import os
import re
from pdf2image import convert_from_path

path = "./"

fname = []
for root,d_names,f_names in os.walk(path):
	for f in f_names:
		fname.append(os.path.join(root, f))

# print("fname = %s" %fname)
for fnam in fname :
	print fnam
	if fnam.endswith('.pdf') :
		pages = convert_from_path(fnam, 500)
		# print len(pages)
		# print re.sub('\.pdf$', '.jpg', fnam)
		jpgnam = re.sub('\.pdf$', '.jpg', fnam)
		pages[0].save(jpgnam, 'JPEG')


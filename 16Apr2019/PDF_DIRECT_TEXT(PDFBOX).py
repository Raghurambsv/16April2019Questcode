import pdfbox
p = pdfbox.PDFBox()
text = p.extract_text('/Users/raghuram.b/Desktop/junk/pdf/BIRF-ZZ-MDZZZ-510-0007.pdf')
print(text)
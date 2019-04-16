 
import glob2



ocr_path='/Users/raghuram.b/Desktop/Exxon/Self_OCR/TXTs/'

filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/LATEST(Jan 15)/PDFs/*")
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
  




#single file only
#################
#
#import PyPDF2
#import glob2
#
#
#content = ""
#ocr_path='/Users/raghuram.b/Desktop/'
#
#filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/LATEST(Jan 15)/PDFs/BIRF-AH-VQULT-800-1005.pdf")
#for path in filelist:
#    pdf=PyPDF2.PdfFileReader(path,'rb')
#    pages=pdf.getNumPages()
#    for i in range(0,pages):    
#        content += pdf.getPage(i).extractText() + "\n"
#        ext=path.split('/')[-1] 
#        ext=ext.replace("pdf","txt") 
#        ext=ext.replace("PDF","txt")
#        with open(ocr_path+ext,mode='w') as file:
#            file.write(content+"\n")
#  
#

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

with open("/Users/raghuram.b/Desktop/BIRF-EM-DCCAL-510-0003.PDF","rb") as in_f:
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
        

 
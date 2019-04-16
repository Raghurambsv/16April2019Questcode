#!/bin/sh
echo "TRAIN DATA FOLDER STARTED"
files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DISO`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/DISO/
fi
done
done

echo "DISO ENDED"
files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DLAY`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/DLAY/
fi
done
done

echo "DLAY ENDED"
files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DPID`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/DPID/
fi
done
done

echo "DPID ENDED"
files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DSHP`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/DSHP/
fi
done
done

echo "DSHP ENDED"
	      
files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i OTHERS`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/OTHERS/
fi
done
done

files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i RCRI`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/RCRI/
fi
done
done

files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i RTQM`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/RTQM/
fi
done
done

files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i RZZZ`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/RZZZ/
fi
done
done

files=`cat /Users/raghuram.b/Desktop/train_filelist.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i XXER`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/train/XXER/
fi
done
done

#Validation folder starts

echo "TRAIN DATA FOLDER ENDED"

files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DISO`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/DISO/
fi
done
done

echo "DISO ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DLAY`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/DLAY/
fi
done
done

echo "DLAY ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DPID`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/DPID/
fi
done
done

echo "DPID ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i DSHP`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/DSHP/
fi
done
done


	      
echo "DSHP ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i OTHERS`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/OTHERS/
fi
done
done

echo "OTHERS ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i RCRI`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/RCRI/
fi
done
done

echo "RCRI ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i RTQM`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/RTQM/
fi
done
done
echo "RTQM ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i RZZZ`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/RZZZ/
fi
done
done

echo "RZZZ ENDED"
files=`cat /Users/raghuram.b/Desktop/validation_list.txt`
for i in $files
do
category=`ls /Users/raghuram.b/Desktop/A4_SINGLEPAGE|grep -i XXER`
for c in $category
do 
if [ $i = $c ]
then
     cp /Users/raghuram.b/Desktop/A4_SINGLEPAGE/$c /Users/raghuram.b/Desktop/data/validation/XXER/
fi
done
done
echo "XXER ENDED"
echo "PROGRAM ENDED"

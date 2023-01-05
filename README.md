This is a repository for atlas construction of medical images. The dataset is in-house 125 CT images of heart. The idea is to construct an atlas with registering a set of images togheter to create an average CT heart image. Then I extracted an aortic heart segmentation from constructed atlas with desired image procesing program. For label in this repository, I used Slicer 5.0.3, using tresholding and then fine cutting segmentation by hand. This segmentation is then used to acqure a mask of any given image. We register atlas to given image and then use the same transformation to transform the atlas mask to desired mask. Lastly we transform some landmarks and evaluate how well the transformed landmarks are predicted.

We construct and atlas by running example.py, which is devided in cells
1st cell is registration procedure of and image to atlas.
2nd cell is construction of atlas.
3rd cell is statistic for atlas point detection 

In folder images we have given 3 .nii images used in example.
In folder AtlasImage the computed atlas and label are saved, along with the computed atlas from example.
In folder points, we have 3 corresponding .txt to the original landmarks and their corresponding csv file point.csv, in points_transformed.csv are the transformed landmarks from atlas.

AtlasRegistration.py is a python script with the development code inside, unnecesary to run, this is for developing purposes as backup.

Repository was run on python 3.9.15.
Gpu: nvidia rtx 3070 Ti
Cpu: AMD Ryzen 7 5800X

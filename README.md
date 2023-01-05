This is a repository for atlas construction of medical images. The dataset is in-house 125 CT images of heart. We construct and atlas by running example.py, which is devided in cells
1st cell is registration procedure of and image to atlas.
2nd cell is construction of atlas.
3rd cell is statistic for atlas point detection 

In folder images we have given 3 .nii images used in example.
In folder AtlasImage the computed atlas and label are saved, along with the computed atlas from example.
In folder points, we have 3 corresponding .txt to the original landmarks and their corresponding csv file point.csv, in points_transformed.csv are the transformed landmarks from atlas.

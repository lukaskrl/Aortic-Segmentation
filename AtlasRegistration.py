#%%
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
#%%

# The atlas and associated segmentation is loaded once and held in memory
N = 25
movingImage = sitk.ReadImage('AtlasRegistration/registered_registeredTo101.nii')
movingLabel = sitk.ReadImage('AtlasRegistration/Registered101_label.nii')
fixedImage = sitk.ReadImage('AorticLandmarkSegmentation/spine_localization_dataset/images/{}.nii'.format(N))


plt.figure()
plt.title('Moving image')
plt.imshow(sitk.GetArrayFromImage(movingImage)[:,200,:], cmap='gray', origin='lower')
plt.figure()
plt.title('fixed image')
plt.imshow(sitk.GetArrayFromImage(fixedImage)[:,200,:], cmap='gray', origin='lower')

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('translation'))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('nonrigid'))
elastixImageFilter.


elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(movingImage)
elastixImageFilter.Execute()
transformParameterMap = elastixImageFilter.GetTransformParameterMap()


resultImage = elastixImageFilter.GetResultImage()
plt.figure()
plt.title('Result image')
plt.imshow(sitk.GetArrayFromImage(fixedImage)[:,200,:], cmap='gray', origin='lower')
plt.imshow(sitk.GetArrayFromImage(resultImage)[:,200,:], cmap='gray', origin='lower', alpha=0.5)

sitk.WriteImage(resultImage, 'AtlasRegistration/registered.nii')

resultLabel = sitk.Transformix(movingLabel, transformParameterMap)
plt.figure()
plt.title('Result label')

plt.imshow(sitk.GetArrayFromImage(resultLabel)[:,200,:], cmap='gray', origin='lower')
sitk.WriteImage(resultLabel, 'AtlasRegistration/registered_label.nii')

#%% get points on atlas 
N = 101
fixedImage = sitk.ReadImage('AtlasRegistration/registered_registeredTo101.nii')
movingImage = sitk.ReadImage('AorticLandmarkSegmentation/spine_localization_dataset/images/{}.nii'.format(N))

plt.figure()
plt.title('Moving image')
plt.imshow(sitk.GetArrayFromImage(movingImage)[:,200,:], cmap='gray', origin='lower')
plt.figure()
plt.title('fixed image')
plt.imshow(sitk.GetArrayFromImage(fixedImage)[:,200,:], cmap='gray', origin='lower')

# elastixImageFilter = sitk.ElastixImageFilter()
# elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('translation'))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('nonrigid'))


# elastixImageFilter.SetFixedImage(fixedImage)
# elastixImageFilter.SetMovingImage(movingImage)
# elastixImageFilter.Execute()

# resultImage = elastixImageFilter.GetResultImage()
# transformParameterMap = elastixImageFilter.GetTransformParameterMap()

path_to_landmarks = os.path.join(os.getcwd(), 'AorticLandmarkSegmentation/spine_localization_dataset/aorta_setup/points_world_cor.csv')
f = np.loadtxt(path_to_landmarks, delimiter=',')
index = np.where(f[:,0] == N)[0][0]


#%%
#points on original image
ref_point = [f[index][1],f[index][2],f[index][3]]
print(ref_point)
ref_point = movingImage.TransformPhysicalPointToIndex(ref_point)
print(ref_point)

ref_point = [f[index][4],f[index][5],f[index][6]]
print(ref_point)
ref_point = movingImage.TransformPhysicalPointToIndex(ref_point)
print(ref_point)

ref_point = [f[index][7],f[index][8],f[index][9]]
print(ref_point)
ref_point = movingImage.TransformPhysicalPointToIndex(ref_point)
print(ref_point)

ref_point = [f[index][10],f[index][11],f[index][12]]
print(ref_point)
ref_point = movingImage.TransformPhysicalPointToIndex(ref_point)
print(ref_point)

ref_point = [f[index][13],f[index][14],f[index][15]]
print(ref_point)
ref_point = movingImage.TransformPhysicalPointToIndex(ref_point)
print(ref_point)

ref_point = [f[index][16],f[index][17],f[index][18]]
print(ref_point)
ref_point = movingImage.TransformPhysicalPointToIndex(ref_point)
print(ref_point)


image = sitk.GetArrayFromImage(movingImage)
plt.figure()
implot = plt.imshow(image[:, ref_point[1], :], cmap='gray', origin='lower')
plt.scatter(ref_point[0], ref_point[2], c='r', s=40)

plt.figure()
implot = plt.imshow(image[ref_point[2], :, :], cmap='gray', origin='lower')
plt.scatter(ref_point[0], ref_point[1], c='r', s=40)

plt.figure()
implot = plt.imshow(image[:, :, ref_point[0]], cmap='gray', origin='lower')
plt.scatter(ref_point[1], ref_point[2], c='r', s=40)

image = sitk.GetArrayFromImage(fixedImage)
plt.figure()
implot = plt.imshow(image[:, ref_point[1], :], cmap='gray', origin='lower')
plt.scatter(ref_point[0], ref_point[2], c='r', s=40)

plt.figure()
implot = plt.imshow(image[ref_point[2], :, :], cmap='gray', origin='lower')
plt.scatter(ref_point[0], ref_point[1], c='r', s=40)

plt.figure()
implot = plt.imshow(image[:, :, ref_point[0]], cmap='gray', origin='lower')
plt.scatter(ref_point[1], ref_point[2], c='r', s=40)



print(fixedImage.GetOrigin(), fixedImage.GetSpacing())
print(movingImage.GetOrigin(), movingImage.GetSpacing())
print(resultImage.GetOrigin(), resultImage.GetSpacing())

plt.figure()
plt.title('resultimage')
plt.imshow(sitk.GetArrayFromImage(resultImage)[:,200,:], cmap='gray', origin='lower')
#%%
# transform points to fixed image

transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetTransformParameterMap(transformParameterMap)
transformixImageFilter.SetMovingImage(fixedImage)
transformixImageFilter.SetFixedPointSetFileName('AtlasRegistration/o_n101.txt')
transformixImageFilter.SetOutputDirectory('AtlasRegistration')
transformixImageFilter.Execute()
#%%
# print transformed points

image = sitk.GetArrayFromImage(fixedImage)
# ref_point = [309, 96, 5167]
ref_point =  [32.242120, -198.147878,2168.709892 ]
ref_point = resultImage.TransformPhysicalPointToIndex(ref_point)
print(ref_point)

plt.figure()
implot = plt.imshow(image[:, ref_point[1], :], cmap='gray', origin='lower')
plt.scatter(ref_point[0], ref_point[2], c='r', s=40)

plt.figure()
implot = plt.imshow(image[ref_point[2], :, :], cmap='gray', origin='lower')
plt.scatter(ref_point[0], ref_point[1], c='r', s=40)

plt.figure()
implot = plt.imshow(image[:, :, ref_point[0]], cmap='gray', origin='lower')
plt.scatter(ref_point[1], ref_point[2], c='r', s=40)




#%%
# save 3 values into array and insert into points array
points = []
for i in range(1, f.shape[1], 3):
    single = []
    for j in range(3):
        single.append(f[index][i+j])
    points.append(single)
print(points)


path_to_points = os.path.join(os.getcwd(), 'AtlasRegistration/points.pts')
#transform points to atlas
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetTransformParameterMap(transformParameterMap)
transformixImageFilter.SetMovingImage(fixedImage)
transformixImageFilter.SetFixedPointSetFileName(path_to_points)
transformixImageFilter.SetOutputDirectory('AtlasRegistration')
transformixImageFilter.Execute()


# transform Landmarks to atlas space


#%%
## crop movingImage by movingLabel
# movingImage = sitk.ReadImage('AorticLandmarkSegmentation/spine_localization_dataset/images/1.nii')
# movingLabel = sitk.ReadImage('AtlasRegistration/Atlas_label_1.nii')
movingImage = sitk.Mask(movingImage, movingLabel)
plt.figure()
plt.title('Moving image')
plt.imshow(sitk.GetArrayFromImage(movingImage)[:,200,:], cmap='gray', origin='lower')

#%% construct atlas from 125 images

path_to_all_nifti = os.path.join(os.getcwd(), 'AorticLandmarkSegmentation/spine_localization_dataset/images')
path_to_atlas = os.path.join(os.getcwd(), 'AorticLandmarkSegmentation/spine_localization_dataset/images/101.nii')

a = np.array([])
atlas = sitk.ReadImage(path_to_atlas)
# type to float
atlas = sitk.Cast(atlas, sitk.sitkFloat32)
atlasArray = []
pointsArray = []

for idx, file in enumerate(os.listdir(path_to_all_nifti)):

    s = ''.join(x for x in file if x.isdigit())

    img = sitk.ReadImage(os.path.join(path_to_all_nifti, file))

    #stack images into array
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('translation'))
    elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
    elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('nonrigid'))
    elastixImageFilter.SetFixedImage(atlas)
    elastixImageFilter.SetMovingImage(img)
    elastixImageFilter.Execute()

    resultImage = elastixImageFilter.GetResultImage()
    print('result image shape: ', resultImage.GetSize())
    print('atlas image shape: ', atlas.GetSize())
    print('result type: ', resultImage.GetPixelIDTypeAsString())
    print('atlas type: ', atlas.GetPixelIDTypeAsString())
    atlas += resultImage

    # Save transformed points of each image to atlas
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetMovingImage(img)
    transformixImageFilter.SetFixedPointSetFileName('AtlasRegistration/atlasPoints/points{}.txt'.format(s))
    transformixImageFilter.SetOutputDirectory('AtlasRegistration/final')
    transformixImageFilter.Execute()

atlas /= 125
#%%
plt.figure()
plt.title('Atlas image') 
plt.imshow(sitk.GetArrayFromImage(atlas)[:,220,:], cmap='gray', origin='lower')
sitk.WriteImage(atlas, 'AtlasRegistration/registered_registeredTo101.nii')
# plt.imshow(np.mean(atlas, axis=3)[:,200,:], cmap='gray', origin='lower')
#%%
path_to_atlas = os.path.join(os.getcwd(), 'AorticLandmarkSegmentation/spine_localization_dataset/images/101.nii')
img = sitk.ReadImage(path_to_atlas)
img = sitk.Cast(img, sitk.sitkFloat32)
plt.figure()
plt.title('Atlas image')
plt.imshow(sitk.GetArrayFromImage(img)[:,220,:], cmap='gray', origin='lower')


#%% get statistics of all points
path_to_landmarks = os.path.join(os.getcwd(), 'AorticLandmarkSegmentation/spine_localization_dataset/aorta_setup/points_world_cor.csv')
path_to_final  = os.path.join(os.getcwd(), 'AtlasRegistration/final/points.csv')


allpoints = []

f = np.loadtxt(path_to_landmarks, delimiter=',')
g = np.loadtxt(path_to_final, delimiter=',')
#calculate eculidean error

errors = []
for i in range(f.shape[0]):
    for j in range(1, f.shape[1], 3):
        single = []
        single1 = []
        for k in range(3):
            single.append(f[i][j+k])
            single1.append(g[i][j+k])
        errors.append(np.linalg.norm(np.array(single) - np.array(single1)))

print('mean error: ', np.mean(errors))
print('std error: ', np.std(errors))

        

    










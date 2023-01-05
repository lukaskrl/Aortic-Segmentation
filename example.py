#%%
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np

#%% Register atlas to desired image
N = 2
movingImage = sitk.ReadImage('AtlasImage/registered_registeredTo101.nii')
movingLabel = sitk.ReadImage('AtlasImage/Registered101_label.nii')
fixedImage = sitk.ReadImage('Images/{}.nii'.format(N))


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
elastixImageFilter.SetOutputDirectory('transforms')


elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(movingImage)
elastixImageFilter.Execute()

resultImage = elastixImageFilter.GetResultImage()
#%% Plot results
plt.figure()
plt.title('fixed image')
plt.imshow(sitk.GetArrayFromImage(fixedImage)[:,200,:], cmap='gray', origin='lower')

plt.figure()
plt.title('Result transformed atlas image')
plt.imshow(sitk.GetArrayFromImage(resultImage)[:,200,:], cmap='gray', origin='lower', alpha=0.5)
transformParameterMap = elastixImageFilter.GetTransformParameterMap()

# transform segmentation of atlas by same transformation to desired image
resultLabel = sitk.Transformix(movingLabel, transformParameterMap)
plt.figure()
plt.title('Result label')
plt.imshow(sitk.GetArrayFromImage(resultLabel)[:,200,:], cmap='gray', origin='lower')

#%% Construction of atlas

path_to_all_nifti = os.path.join(os.getcwd(), 'images')
path_to_atlas = os.path.join(os.getcwd(), 'images/3.nii')

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
    elastixImageFilter.SetOutputDirectory('transforms')
    elastixImageFilter.Execute()

    resultImage = elastixImageFilter.GetResultImage()
    print('result image shape: ', resultImage.GetSize())
    print('atlas image shape: ', atlas.GetSize())
    print('result type: ', resultImage.GetPixelIDTypeAsString())
    print('atlas type: ', atlas.GetPixelIDTypeAsString())
    atlas += resultImage

    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    # Save transformed points of each image to atlas
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetMovingImage(img)
    transformixImageFilter.SetFixedPointSetFileName('points/o_n{}.txt'.format(s))
    transformixImageFilter.SetOutputDirectory('points/transformed/{}'.format(s))
    transformixImageFilter.Execute()

atlas = sitk.GetArrayFromImage(atlas)
atlas = atlas / len(os.listdir(path_to_all_nifti))
atlas = sitk.GetImageFromArray(atlas)
sitk.WriteImage(atlas, 'AtlasImage/atlas.nii')

#%% Point statistics

path_to_landmarks = os.path.join(os.getcwd(), 'points/points.csv')
path_to_final  = os.path.join(os.getcwd(), 'points/points_transformed.csv')
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
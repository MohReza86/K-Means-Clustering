''' @uthor: Mohammadreza Baghery '''

''' This module takes in background-subtracted images to segment cell bodies and extract a variety of data for each detected cell.
It includes a contrast enhancement algorithm which is useful for images with low fluorescence intensity.
The contrast-enhaced images are then fed to a K-means clustering segmentation to identify cells from background and artifacts.
Then a binary mask is applied on the segmented image to isolate the cell body cluster as the selected cluster. After removing
non-cell objects, the binary mask image is subjected to morphological operations to correct the shape of erroneously-developed
cell body masks. Next, the contours of the segmented cell bodies are identified and a heat map of fluorescence intensity
variation is developed. Finally the algorithm outputs all the processed images and CSV files containing the number of detected cells,
fluorescence intensity, area, height and width coordinate, and Euclidean distance to neighboring cells for each detected cell. '''


# importing the parameters module
from parameters import*

# importing the required libraries 
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt 
from skimage import measure 
from skimage.color import rgb2gray, label2rgb
from sklearn.neighbors import NearestNeighbors  
import pandas as pd 
import csv
from skimage.measure import label, regionprops
from skimage import data, filters, measure, morphology
from skimage.feature import canny 
from skimage import io, img_as_float, img_as_ubyte


####################### PARRT 1: IMAGE SEGMENTATION #######################

# Reading the original image from the source path
def read_img(path):
    return cv2.imread(path) 

original_image = read_img(input_path)  

# converting the image to float
original_image_float = img_as_float(original_image)

# converting the original image to gray value
original_image_gray = rgb2gray(original_image)

image = Image.open(input_path)

def contrast_enhancemenet(image, con_factor):

    ''' Returns the image with increased contrast 
    by a given factor 

    image: raw image
    con_factor: contrast factor  '''

    #image brightness enhancer
    enhancer = ImageEnhance.Contrast(image)
    img_output = enhancer.enhance(con_factor)
    return img_output

contrasted_image = contrast_enhancemenet(image, contrast_factor)

# height and width of the image
height, width = original_image_gray.shape


# displaying the contrasted image
plt.imshow(contrasted_image, cmap = 'gray')
plt.title('Contrast-Enhanced Image')
plt.show()


def K_means_clustering(contrasted_image, original_image, k=4): 

    ''' Returns segmented image based on K-means clustering. 
        K-means clustering is an unsupervised machine learning algorithm 
        that aims to partition N observations into K clusters in which 
        each observation belongs to the cluster with the nearest mean.

        contrasted_image: this contrast-enhanced image is used for detectig the cluster
        original_image: the cluster masks are overlaid on the original image
        k = number of clusters (set to 4 by default)

     '''

    img_array = np.array(contrasted_image)
    # convert to RGB 
    img_RGB = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # Convert MxNx3 image into Kx3 where K=MxN
    img_reshape = img_RGB.reshape((-1,3))
    # we convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    img_float = np.float32(img_reshape)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Number of attempts refer to number of times algorithm is executed using different initial labelings
    attempts = 10
    # specify how initial seeds are taken
    _,labels,(centers) = cv2.kmeans(img_float, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # convert back to 8 bit values 
    centers = np.uint8(centers) 
    # flatten the labels array 
    labels = labels.flatten()

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(original_image.shape)
    segmented_image_gray = rgb2gray(segmented_image)
    return segmented_image_gray

# the segmented image including all detected clusters 
segmented_image = K_means_clustering(contrasted_image, original_image_float)

# diplaying the segmented image
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.show()

# threshold pixel value for the desired cluster (i.e. cell body); this value is specified after viewing the segmented image
threshold = 0.5

# applying a binary mask to isolate the cell body cluster as the selected cluster 
label_image = measure.label(segmented_image > threshold, connectivity = original_image_gray.ndim)



####################### PARRT 2: POST-PROCESSING #######################

# Step 1: Removing objects with sizes less than the cell type of interest from the labeled image
if ChAT:
    mask = morphology.remove_small_objects(label_image, min_cell_size_ChAT)
elif PV:
    mask = morphology.remove_small_objects(label_image, min_cell_size_PV)
elif nNos:
    mask = morphology.remove_small_objects(label_image, min_cell_size_nNos)

# Step 2: Applying morphological closing algorithm as to correct the shape of the detected labels 
mask = morphology.closing(mask)

# overlaying the mask (i.e., processed detected cell bodies) on the original image
#mask_img_overlay = label2rgb(mask, image=original_image, bg_label=0)



####################### PARRT 3: PROPERTIES OF SEGMENTED CELLS  #######################
 
# defining the properites of the segmented image
props = measure.regionprops_table(mask, original_image_gray, properties = ['label', 'area', 'centroid', 'mean_intensity'])

# converting the defined properites into pandas dataframe
df = pd.DataFrame(props)

label = df['label']

intensity = df['mean_intensity'] 

mean_intensity = intensity.mean()

area = df['area']

x = df['centroid-1'].tolist()
y = df['centroid-0'].tolist()

x_integ = [i for i in x x_integ.append(int(i))]

y_integ = [i for i in y y_integ.append(int(i))]

# coordinates of the detected cells 
coordinates = list(zip(x_integ, y_integ))

intensity = intensity.tolist()
area = area.tolist()

cell = list(range(1, (len(y_integ)+1)))

# normalizing the intensity values from 0 to 1; this normalization is used for develping the intensity heat map 
def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norm_inten = norm_data(intensity)

new_norm_inten = [i for i in norm_inten new_norm_inten.append(abs(i-1))]

original_image = read_img(input_path)  

# Developing the heatmap image based on the intensity of detected cells 
for index, value in enumerate(coordinates):
    heat_map = cv2.circle(original_image, value, 4, (norm_inten[index]*255, norm_inten[index]*255, new_norm_inten[index]*255), 30)


# removing holes in the object for edge detection
mask = morphology.remove_small_holes(mask, 15)

# processed mask image
plt.imshow(mask, cmap = 'gray')
plt.title('Processed Mask Image')
plt.show()


original_image = read_img(input_path)  

# drawing circles over the detected cells
for i in coordinates:
    cells_marked = cv2.circle(original_image, i, 25, (255, 0, 0), 4)

# displaying the marked cells 
plt.imshow(cells_marked)
plt.title('Marked Cells')
plt.show()

# preparing the mask image for contour detection 
def img_float(img):
    return img_as_float(img)

mask_float = img_float(mask)

# contour detection 
edge_image = canny(mask_float, sigma = 0.5)

# overlaying the detected cell contours on the original image 
edge_overlay = label2rgb(edge_image, image=original_image_gray, bg_label=0, colors = ['red'])

# displaying the overlay image of detected cell contours
plt.imshow(edge_overlay)
plt.title('Cell Contour')
plt.show()

# Displaying the heat map 
plt.imshow(heat_map)
plt.title('Heat Map')
plt.show()


####################### PARRT 4: SAVING OUTPUT IMAGES AND WRITING THE DATA TO CSV FILES #######################

# saving the image with marked detected cells 
plt.imsave(f"{output_path}/{marked_cells}{output_image_format}", cells_marked)

# saving the heat map of cells intensity 
plt.imsave(f"{output_path}/{heat_map_img}{output_image_format}", heat_map)


# nearest neighbors 
nearest_neighbor = NearestNeighbors(n_neighbors = num_neighbors, metric = 'euclidean')
nearest_neighbor.fit(coordinates)
dist, indices = nearest_neighbor.kneighbors(coordinates)

# writing the cell coordinate and intensity to a CSV file
intensity_header = ['Cell', 'Width Coordinate', 'Height Coordinate', 'Area', 'Intensity'] 
filename_intensity = f"{output_path}/{intensity_csv}.csv"

with open(filename_intensity, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(intensity_header)

        for w in range(len(label)):
            wr.writerow([cell[w], x_integ[w], y_integ[w], area[w], intensity[w]])


# writing the cell coordinate and nearest neighbors to a CSV file
header_nearest_neighbor = ['Cell', 'Width Coordinate', 'Height Coordinate', 'Dist_1', 'Dist_2', 'Dist_3', 'Dist_4', 
'Dist_5', 'Dist_6', 'Dist_7', 'Dist_8', 'Dist_9', 'Dist_10']

filename_nearest_neighbor = f"{output_path}/{nearest_dist_csv}.csv"

with open(filename_nearest_neighbor, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(header_nearest_neighbor)

        index = 0
        for w in dist: 
            wr.writerow([cell[index]] +  [x_integ[index]] + [y_integ[index]] + w[1:].tolist())
            index = index + 1








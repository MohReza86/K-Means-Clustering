''' @uthor: Mohammadreza Baghery '''

''' The parameters module specifies all the constants to be used in the k_means_clustering module. '''

# path to the original image 
input_path = ''

# output path folder 
output_path = ''

# name of the processed image to be saved 
image_name = ''

# format of output images
output_image_format = '.tiff'

# factor below 1 makes the image greyer, while factor above 1 increases the contrast of the image
contrast_factor = 1.0

# number of clusters (k) 
k = 4

# number of nearest neighbors to each cell 
num_neighbors = 11

# specigying the cell type 
ChAT = True
PV = False
nNos = False

# names of output images
segmented_img = f"segmented_img_{image_name}"
marked_cells = f"marked_cells_{image_name}"
heat_map_img = f'heat_map_{image_name}'
intensity_csv = f"intensity_{image_name}"
nearest_dist_csv = f"nearest_dist_{image_name}"

# minimum size (pixels) of the segmented objects to be recognized as full-sized cells 
min_cell_size_ChAT = 80
min_cell_size_PV = 50
min_cell_size_nNos = 0


# K-means-Clustering
Using k-means clustering algorithm, the main module segments cell bodies in fluorescent images.


The  algorithm  is  divided  into  two  modules,  parameters.py  and  k_means_clustering.py. 
The  parameters  module  requires  the  user  to  provide  basic  constant  information, 
notably the path to a background-subtracted image and the path to a folder to output 
processed images and CSV files. Further, it requires the user to specify the type of the 
cell (ChAT, PV or nNos) to be analyzed by providing True or False. The contrast-factor 
(set  to  1  by  default)  is  used  in  the  case  of  images  with  low  intensity  and  it  asks  the 
degree to which the image needs to be enhanced. In addition, the ‘k’ is the number of 
clusters  to  which  that  the  algorithm  attempts  to  segment  the  image.  The  user  can 
modify this value if the segmentation image is not desirable. 

The k_means_clustering module consists of the required functions and 
algorithms for image segmentation and post-processing. Upon reading the image, the 
algorithm initially performs the contrast enhancement. This image is then fed to the k-
means clustering algorithm for image segmentation. Then a binary mask is applied on 
the  segmented  image  to  isolate  the  cell  body  cluster  as  the  selected  cluster.  After 
removing non-cell objects, the processed binary mask image is subjected to 
morphological  operations  to  correct  the  shape  of  erroneously-developed  cell  body 
masks. Next, the contours of the segmented cell bodies are identified and a heat map 
of  fluorescence  intensity  variation  is  developed.  Finally  the  algorithm  outputs  all  the 
processed images and CSV files containing the number of detected cells, fluorescence 
intensity,  area,  height  and  width  coordinate,  and  Euclidean  distance  to  neighboring 
cells for each detected cell. 

The user is required to provide the the necessary information on parameters.py 
and  execute  the  k_means_clustering.py.  Upon  running,  the  algorithm  displays  the 
processed  images  to  the  user  after  the  completion  of  each  image  processing  step, 
namely contrast-enhancement, image segmentation, cell body masks, cell body 
contours and heat map. 

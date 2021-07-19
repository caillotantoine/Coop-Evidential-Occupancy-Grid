# upstream-coop-loc
The goal of this project is to locate vehicles at an intersection using the data of vehicles streamed to the infrastructure to locate everyone. 


# Documentation
## ROS Launch
### bbox_gol_test
Runs 
- position of : car 1
- position of : car 2
- position of : car 3
- GOL_publisher

### gol_gog

### simple_test

## ROS Nodes

### GOL_publisher
This node takes the positions of the cars available and draw theme on the image
#### Publish
- Image (with 2D BBox drawn)
#### Subscribe
- Image (from camera)
- BBox 3D

### gndTruthCAR
#### Publish
- BBox 3D
#### Subscribe
Nothing
#### Param
- Path to the json of the vehicle information

## Classes & functions

### BBoxManager
#### drawBoxes
Draw 3D Bounding box on the image.
##### IN
- image
- BBox3D
##### OUT
- image with 3D bbox drawn

#### draw2DBoxes
Draw 2D Bounding box on the image.
##### IN
- image
- BBox3D
##### OUT
- image with 2D bbox drawn
- list of 2D bbox
	- list of 2D points on the image (4 elements, one per corner). Note that everything remains drawn, even if not on the images. 
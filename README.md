# upstream-coop-loc
This project use the 2 dimenssional bounding boxes from cameras of road side units (RSU) and on-board units (OBU) to generate an [occupancy grid map](#evidential-occupancy-grid) or a [semantic occupancy map](#evidential-semantic-occupancy-grid) of a scene. We focused our work on situations at intersection and especially at round-abouts. 

<!-- The first itteration of this project is available at [Evidential Occupancy Grid](#evidential-occupancy-grid) section. The second itteration featuring semantic aspect is available at [Evidential Semantic Occupancy Grid](#evidential-semantic-occupancy-grid) section. -->

## Evidential Occupancy Grid
This first iteration of the project is based on ROS. However, ROS induce stability issues because of the parallel runing process desynchronising. Thus, the second iteration of the project has been developped in Python / C++ solely. 


## Evidential Semantic Occupancy Grid


### Location of the source code
The code for this part of the project is located in the folder `standalone_project/full_project`.

### Source files 

#### testbench.py
Testbench is the entry point of the project. It parses the command line, and link the blocks together to for the pipeline. The reading of the dataset is also done in this file as well as the recordings and the GUI management. 

#### EGG.py
Evidential Grid Generator create an evidential map container and provide back-projection and rasterisation tools. Noise of position is applied in this file. 

#### projector.py
This file host the back-projection code based on Plücker's coordinates. It also hosts tools to manage matrix, to display in open3D the projection, and a filter to ensure the bounding box is well fitted in the image. `project_BBox2DOnPlane` is the most important function to seek.


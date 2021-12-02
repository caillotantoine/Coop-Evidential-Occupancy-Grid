# Cooperative evidential semantic grid generation

## Structure
In the scene, several agents are available, perceiving the scene. Each of them transmits a set of 2D bounding box of detected objects (associated with their label) as well as the sensor's information estimated in the scene (position and camera matrix). This information is used to create an evidential semantic grid. There is one evidential semantic grid per agents which are all merged before taking the decision for each cell in the final semantic grid.

## Classes
We manage 3 classes:
- Vehicles
- Pedestrians
- Terrain (free space)

These 3 classes give us a discernment frame Ω = {Ø, V, P, T}.\
Focal elements are defined as follows: 
2^Ω = {Ø, V, P, T, {V, P}, {V, T}, {P, T}, Ω}.


## Agent
To be explained.


## Evidential semantic grid
The format of the grid is a `Tensor<x, y, z>` in which `<x, y>` give the position on the map (in cell coordinate) and `<z>` gives the focal element.

The map is defined by:
- it's size `mapsize` given in meters
- grid's size `gridsize` given in cell number

The size of a cell is given as follow : `cellsize = mapsize / gridsize`.


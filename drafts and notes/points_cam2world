# FROM INFRA

#C = [-7.297264496026017   -3.3409162752890182   60.83984375   1.0] # in meters @ frame 61

Vx = [-3.1178733516979804   -0.15989094111272095  55.322265625      1.0] # in meters @ frame 191
Vy = [5.990089019598987     4.1563882993135834    28.1982421875     1.0] # in meters @ frame 191
Vz = [-10.621881209356935   4.241855158282153     23.86474609375    1.0] # in meters @ frame 191

# GND TRUTH : 
V0 = [11.1   21.5    0.0    1.0]
V1 = [ 1.9   35.2    0.0    1.0]
V2 = [-5.6   23.18   0.0    1.0]

dimensions = [1.8843283653259277   1.5686503648757935   4.5848188400268555] #width, height, length in meters

orientation = 4.584731829125722 # rad from x on axis z

# from infra
T_WCw = [-4.10752676316406e-08, -1.0, -1.4950174431760388e-08, 0.0; 0.9396926164627075, -4.371138828673793e-08, 0.3420201241970062, 0.0; -0.3420201241970062, -0.0, 0.9396926164627075, 13.0; 0.0, 0.0, 0.0, 1.0]

T_CCw = [0   1   0   0; 0   0  -1   0; 1   0   0   0; 0   0   0   1]

disp("Found V1 at")
vx = T_WCw * T_CCw' * Vx'
disp("Distance = ")
sqrt(sum((vx-V1').^2))

disp("Found V2 at")
vy = T_WCw * T_CCw' * Vy'
disp("Distance = ")
sqrt(sum((vy-V2').^2))

disp("Found V0 at")
vz = T_WCw * T_CCw' * Vz'
disp("Distance = ")
sqrt(sum((vz-V0').^2))
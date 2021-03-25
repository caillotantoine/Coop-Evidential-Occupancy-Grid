#ifndef GOL_proj
#define GOL_proj

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

typedef struct {
    float a;
    float b;
    bool vertical;

    Vector2d start, end;
    // y = ax+b
    // if vert, x = a and b is ignored
} line2D;

#endif
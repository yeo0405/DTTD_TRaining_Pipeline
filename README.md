# Pose Trainer

## ply,xyz file get

1. meshlab 
ply
xyz 
    Filters
        → Sampling
        → Poisson-disk Sampling


## Preprocess

1. change the path of dataset in prepocess.sh

2. prepare a folder `objects` which have fodler inside with name of class and .ply , .xyz file like

    ```
    objects
    -cartridge
        -cartridge.ply
        -points.xyz
    ```

3. `bash preprocess/preprpcess.sh`

BLEND
min: [-0.08975 -0.04765 -0.0145 ]
max: [0.08975 0.04765 0.0145 ]
mean: [0. 0. 0.]
mean: [0. 0. 0.]
num vertices: 8

PLY
min: [-0.08975 -0.04765 -0.0145 ]
max: [0.08975 0.04765 0.0145 ]
mean: [-0.00163871 -0.00043377 -0.00012833]
num vertices: 218
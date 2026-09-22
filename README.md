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
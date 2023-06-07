# localthickness
Fast local thickness in 3D and 2D.
Implements the algorithm described in [our CVPR-W (CVMI) paper](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/papers/Dahl_Fast_Local_Thickness_CVPRW_2023_paper.pdf).

## Installation
Install the module using ```pip install localthickness``` or clone the repository.

## Use
``` python
import localthickness as lt

#  Make a binary test volume. 
B = lt.create_test_volume((100, 500, 400), sigma=15, boundary=0.001)

# Compute thickness and separation.
thickness = lt.local_thickness(B, scale=0.5)
separation = lt.local_thickness(~B, scale=0.5)

# Visualize.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(B[10])
ax[1].imshow(thickness[10], cmap=lt.black_plasma())
ax[2].imshow(separation[10], cmap=lt.white_viridis())

```

![](https://github.com/vedranaa/local-thickness/raw/main/mwe_figure.png)


## Paper
The fast local thickness method is described and evaluated in our contribution to the 8th IEEE Workshop on Computer Vision for Microscopy Image Analysis (CVMI), held in conjunction with the CVPR 2023 conference. Please cite our paper if you use the method in your work.

```
@inproceedings{dahl2023fast,
  title={Fast Local Thickness},
  author={Dahl, Vedrana Andersen and Dahl, Anders Bjorholm},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={4335--4343},
  year={2023}
}
```


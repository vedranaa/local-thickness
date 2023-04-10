# localthickness
Fast local thickness in 3D and 2D.

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

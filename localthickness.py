import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.colors
import edt


#%% FUNCTIONS FOR COMPUTING LOCAL THICKNESS IN 2D AND 3D

def local_thickness(B, scale=1, mask=None):
    """
    Computes local thickness in 2D or 3D, using a basic or a scaled approach.
    @author: abda@dtu.dk, vand@dtu.dk
    Arguments: 
        B: binary 2D or 3D image.
        scale: downscaling factor, e.g. 0.5 for halving each dim of the image.
        mask: binary mask of the same size of the image defining parts of the
            image to be included in the computation of the local thickness.
    Returns: Local thickness of the same size as B.
    """
    
    if scale==1:
        return local_thickness_basic(B, mask)
    else:
        return local_thickness_scaled(B, scale, mask)
        

def local_thickness_basic(B, mask=None, given_dist=False):
    """
    Computes local thickness in 2D or 3D (without scaling).
    """
    
    dilate = {2: dilate2d, 3: dilate3d}[B.ndim]
      
    # distance field, if not already given distances
    if not given_dist:
        out = edt.edt(B)
    else:
        out = B
    
    if mask is not None:
        out = out * mask
        
    # iteratively dilate the distance field starting with max value
    for r in range(0, int(out.max())):
        temp = dilate(out)
        change = out > r
        np.copyto(out, temp, where=change)
    return out


def local_thickness_scaled(B, scale=0.5, mask=None):
    """
    Computes local thickness in 2D or 3D using scaled approach.
    """
    
    dilate = {2: dilate2d, 3: dilate3d}[B.ndim]
    
    dim = B.shape  # original image dimension
    dim_s = tuple(int(scale*d) for d in dim)  # dowscaled image dimension
    c = coords(dim, dim_s)

    dist = edt.edt(B)
    
    # downscale (order=0 is nearest-neighbor) and masking
    if mask is None:
        mask_s = None
    else: 
        dist = dist * mask
        mask_s = scipy.ndimage.map_coordinates(mask, c, order=0)
        
    dist_s = scipy.ndimage.map_coordinates(dist, c, order=0) * scale
    
    # compute local thickness for downscaled
    out = local_thickness_basic(dist_s, mask=mask_s, given_dist=True)
    
    # free up some memery (does this make difference?)
    del dist_s
    del mask_s

    # flow-over boundary to avoid blend across boundary, will mask later
    B_s = scipy.ndimage.map_coordinates(B, c, order=0)
    temp = dilate(out)
    out[~B_s] = temp[~B_s]
    
    # free up some memery (does this make difference?)
    del B_s

    # upscale, order=1 is bi-linear
    out = scipy.ndimage.map_coordinates(out, coords(dim_s, dim), order=1)
    out *= (1/scale) 
    out *= B    
    
    # mask output
    if mask is not None:
        out *= mask
    return out


def coords(old, new):
    '''Query coordinates when rescaling the image of shape old to shape new.
       Made to be used with ndimage.map_coordianges for rescaling images.
    '''

    if len(old)==3:
        c = np.mgrid[0 : old[0]-1 : new[0] * 1j, 0 : old[1]-1 : new[1] * 1j, 
                     0 : old[2]-1 : new[2] * 1j]
    elif len(old)==2:
        c = np.mgrid[0: old[0]-1 : new[0] * 1j, 0 : old[1]-1 : new[1] * 1j]
    else:
        return
    
    # we had numerical problems so I make sure that we stay inside the image
    for i in range(len(c)):
        c[i] = np.clip(c[i], 0, old[i] - 1)

    return c


def dilate3d(vol):
    ''' Dilation with 1-sphere approximated with small kernels.'''
    
    IDX = [None] * 3
    
    #  Displacements left-right, to-from, up-down (6 voxels
    #  or a voxel in each cube face)
    IDX[0] = [[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0],
              [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1]]
    
    #  Face-diagonal displacements (4 voxels in each of 3 planes
    #  or a voxel in each cube edge)
    IDX[1] = [[0, -1, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 1, 0],
              [0, 0, 1, 0, 0, -1, 0, 0, 0, -1, 1, 0],
              [1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0],
              [0, -1, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0],
              [0, -1, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0],
              [1, 0, 0, -1, 0, 0, 0, -1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0],
              [0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1],
              [0, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, -1],
              [0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, -1],
              [1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, -1]]
    
    #  Body-diagonal displacements (a voxel in each cube corner)
    IDX[2] = [[0, -1, 0, -1, 0, -1, 1, 0, 1, 0, 1, 0],
              [0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0],
              [1, 0, 0, -1, 0, -1, 0, -1, 1, 0, 1, 0],
              [1, 0, 1, 0, 0, -1, 0, -1, 0, -1, 1, 0],
              [0, -1, 0, -1, 1, 0, 1, 0, 1, 0, 0, -1],
              [0, -1, 1, 0, 1, 0, 1, 0, 0, -1, 0, -1],
              [1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1],
              [1, 0, 1, 0, 1, 0, 0, -1, 0, -1, 0, -1]] 
    
    
    W = [np.sqrt(6) / (np.sqrt(6) + np.sqrt(3) + np.sqrt(2)),
         np.sqrt(3) / (np.sqrt(6) + np.sqrt(3) + np.sqrt(2)),
         np.sqrt(2) / (np.sqrt(6) + np.sqrt(3) + np.sqrt(2))]
    
    d, r, c = vol.shape
    out = np.zeros(vol.shape)  

    for w, idx in zip(W, IDX):
        
        temp = vol.copy()
        for i in idx:        
            temp[i[0]:d+i[1], i[2]:r+i[3], i[4]:c+i[5]] = np.maximum(
                    temp[i[0]:d+i[1], i[2]:r+i[3], i[4]:c+i[5]],
                    vol[i[6]:d+i[7], i[8]:r+i[9], i[10]:c+i[11]])
        
        out += w * temp
            
    return out


def dilate2d(vol):
    ''' Dilation with 1-disc approximated with small kernels.'''
        
    IDX = [None] * 2
    IDX[0] = [[0, -1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, -1, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, -1, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, -1]]
    IDX[1] = [[0, -1, 0, -1, 1, 0, 1, 0],
              [1, 0, 0, -1, 0, -1, 1, 0],
              [0, -1, 1, 0, 1, 0, 0, -1],
              [1, 0, 1, 0, 0, -1, 0, -1]] 
    
    W = [np.sqrt(2) / (1+np.sqrt(2)),
         1 / (1+np.sqrt(2))]
 
    r, c = vol.shape
    out = np.zeros(vol.shape)  

    for w, idx in zip(W, IDX):
        
        temp = vol.copy()
        for i in idx:        
            temp[i[0]:r+i[1], i[2]:c+i[3]] = np.maximum(
                temp[i[0]:r+i[1], i[2]:c+i[3]], vol[i[4]:r+i[5], i[6]:c+i[7]])
        
        out += w * temp
            
    return out


def local_thickness_conventional(B, mask=None, verbose=False):
    """
    Computes local thickness in 2D or 3D using the conventional approach.
    VERY SLOW, NOT TESTED, USE WITH CAUTION!!!!
    THIS IS JUST FOR COMPARISON!!
    @author: abda@dtu.dk, vand@dtu.dk
    Arguments: B - binary 2D or 3D image.
    Returns: Local thickness of the same size as B.
    """
    
    import skimage.morphology  # imported here since not used elsewhere
    
    # distance field
    df = edt.edt(B)
    if mask is not None:
        df = df * mask
    
    # image that will be updated
    out = np.copy(df) 
    
    # iteratively dilate the distance field starting with max value
    for r in range(1, int(df.max()) + 1):
        if verbose:
            print(f'Dilating with radius {r} of {int(np.max(df))}')
        if B.ndim==2:
            selem = skimage.morphology.disk(r)
        elif B.ndim==3:
            selem = skimage.morphology.ball(r)
        df[df < r] = 0
        temp = skimage.morphology.dilation(df, footprint=selem)
        out = np.maximum(out, temp)    
    out *= B
    if mask is not None:
        out *= mask
    return out       



#%% VISUALIZATION FUNCTIONS


def mixed_colors(N, bg=0.5, bg_alpha=0.5):
    '''Colormap with permuted jet colors.'''
    rng = np.random.default_rng(2022)
    colors = plt.cm.jet(np.linspace(0, 1, N))
    colors = rng.permutation(colors)
    colors = np.vstack((np.array([bg] * 3 + [bg_alpha]), colors))
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap


def black_plasma(bg=0, bg_alpha=1):
    ''' Matplotlib colormap often used for local thickness.
        bg: background color (default black).
        bg_alpha: background alpha value (default opaque).
    '''
    colors = plt.cm.plasma(np.linspace(0, 1, 256))
    colors[:1, :] = np.array([bg] * 3 + [bg_alpha])
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap


def white_viridis():
    '''Another colormap useful for showing both thickness and separation.'''
    colors = np.flip(plt.cm.viridis(np.linspace(0, 1, 256)), axis=0)
    colors[:1, :] = np.array([1, 1, 1, 1])
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap


def pl_black_plasma():
    '''Colormap in plotly format'''
    c = black_plasma()(np.linspace(0, 1, 256))[:, 0:3]
    pl_colorscale = []
    for i in range(256):
        pl_colorscale.append([i/255, f'rgb({c[i,0]},{c[i,1]},{c[i,2]})'])
    return pl_colorscale


def view_thickness_slice(z, T, vmax):
    plt.imshow(T[z], cmap=black_plasma(), vmin=0, vmax=vmax, interpolation='nearest')
    plt.title(f'slice z={z}')
    plt.show()


def arrow_navigation(event, z, Z):
    if event.key == "up" or event.key.lower()=='w':
        z = min(z+1, Z-1)
    elif event.key == 'down' or event.key.lower()=='z':
        z = max(z-1, 0)
    elif event.key == 'right' or event.key.lower()=='d':
        z = min(z+10, Z-1)
    elif event.key == 'left' or event.key.lower()=='a':
        z = max(z-10, 0)
    elif event.key == 'pagedown' or event.key.lower()=='x':
        z = min(z+50, Z-1)
    elif event.key == 'pageup' or event.key.lower()=='e':
        z = max(z-50, 0)
    return z


def show_vol(V, cmap=plt.cm.gray, vmin=None, vmax=None): 
    """
    Shows volumetric data for interactive inspection. 
    Whether it works depends on matplotlib backend.
    @author: vand at dtu dot dk
    """
    def update_drawing():
        ax.images[0].set_array(V[z])
        ax.set_title(f'slice z={z}')
        fig.canvas.draw()
 
    def key_press(event):
        nonlocal z
        z = arrow_navigation(event,z,Z)
        update_drawing()
        
    Z = V.shape[0]
    z = (Z-1)//2
    fig, ax = plt.subplots()
    if vmin is None: 
        vmin = np.min(V)
    if vmax is None: 
        vmax = np.max(V)
    
    ax.imshow(V[z], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'slice z={z}')
    fig.canvas.mpl_connect('key_press_event', key_press)
    plt.show()


#%% HELPING FUNCTIONS

def create_test_volume(dim, sigma=7, threshold=0, boundary=0, frame = True, seed = None):
    """ Creates test volume for local thickness and porosity analysis.
    Arguments:
        dim: tuple giving the size of the volume
        sigma: smoothing scale, higher value - smoother objects
        threshold: a value close to 0, larger value - less material (smaller objects)
        boundary: strength of imposing object boundary pulled inwards
        frame: one-voxel frame of False 
    Returns:
        a test volume
    Example use:
        V = create_test_volume((150,100,50), boundary=0.1)
    For images (2D) use: 
        a = create_test_volume((50,50,1), frame=False)[:,:,0]
    Author: vand@dtu.dk, 2019
    """
    
    if len(dim)==3:
        r = np.fromfunction(lambda x, y, z: 
            ((x / (dim[0] - 1) - 0.5)**2 + (y / (dim[1] - 1) - 0.5)**2+
             (z / (dim[2] - 1) - 0.5)**2)**0.5, dim, dtype=int)
    elif len(dim)==2:
        r = np.fromfunction(lambda x, y: 
            ((x / (dim[0] - 1) - 0.5)**2 + (y / (dim[1] - 1) - 0.5)**2)**0.5,
            dim, dtype=int)
                 
    prng = np.random.RandomState(seed) # pseudo random number generator
    V = prng.standard_normal(dim)
    V[r>0.5] -= boundary;
    V = scipy.ndimage.gaussian_filter(V, sigma, mode='constant', cval=-boundary)
    V = V>threshold
    if frame:
        V[[0,-1]] = False
        V[:, [0,-1]] = False
        if len(dim)==3:
            V[:, :, [0,-1]] = False
    return V


#%% 
def prepare_for_saving(thickness, options):

    factor = options.get('factor', 1)
    pad = options.get('pad', True)
    blend = options.get('blend', True)
    dilate = options.get('dilate', True)
    maxval = options.get('maxval', thickness.max())

    if factor>1:
        Z, Y, X = thickness.shape
        thickness = thickness[Z%factor//2::factor, Y%factor//2::factor, 
                              X%factor//2::factor]
    if pad:
        thickness = np.pad(thickness, ((1, 1), ) * 3, constant_values=0)

    # Grayscale volume which will be saved for shape
    gray = ~(thickness>0)  
    if blend:
        gray = edt.edt(gray) - edt.edt(~gray) - gray + 0.5
        gray = np.clip(gray, -25, 25)
        gray = np.exp(0.1 * gray)
        gray = gray / (gray + 1)  # values between 0 and 1 with 0.5 at interface
    
    if dilate:
        thickness = dilate3d(thickness)

    return thickness, gray, maxval



def save_thickness_volumes(thickness, filename_root, **options):
    '''Saves results of local thickness analysis in two volumes suitable for
    visualization in paraview.
    Inputs:
        thickness: local thickness volume, a 3D numpy array, where 0 is 
            bacground and positive values are thickness.
        filename_root: root of the filename, sufixes will be added 
        factor: int, downscaling factor. Default 1 does not downscale.
        pad: bool, whether to pad with background to ensure closed surfaces.
        dilate: bool, whether to dilate colored object 1px to ensure color on 
            surface.
        blend: whether to blend binary values of object representation for a
            smoother surface.
        maxval: thickness which will be shown with last color in cmap.
        paraview_alpha: whether to invert alpha values as paraview expects.
    '''
    import tifffile

    paraview_alpha = options.get('paraview_alpha', True)
    thickness, gray, maxval = prepare_for_saving(thickness, options)

    cmap = black_plasma(bg=0, bg_alpha=1)  # set bg=1 for white which might blend nicer
    
    with (tifffile.TiffWriter(filename_root + '_gray.tif') as gr, 
            tifffile.TiffWriter(filename_root + '_rgba.tif') as rgba):
        for g, t in zip(gray, thickness):
            gr.write((255 * g).astype(np.uint8))
            t_rgba = cmap(t/maxval)
            if paraview_alpha:
                t_rgba[:, :, 3] = 1 - t_rgba[:, :, 3]  # flipping opacity to work with paraview
            rgba.write((255*t_rgba).astype(np.uint8), photometric='rgb', extrasamples = 'ASSOCALPHA') 
            # rgb.write((255*t_rgba[:, :, :3]).astype(np.uint8), photometric='rgb')
    

#%% VTK WRITE FUNCTIONS

def save_gray2vtk(volume, filename, filetype='ASCII', origin=(0,0,0),
                  spacing=(1,1,1), dataname='gray'):
    ''' Writes a vtk file with grayscace volume data.
    Arguments:
       volume: a grayscale volume, values will be saved as floats
       filename: filename with .vtk extension
       filetype: file type 'ASCII' or 'BINARY'. Writing a binary file might not
           work for all OS due to big/little endian issues.
       origin: volume origin, defaluls to (0,0,0)
       spacing: volume spacing, defaults to 1
       dataname: name associated with data (will be visible in Paraview)
    Author:vand@dtu.dk, 2019
    '''
    with open(filename, 'w') as f:
        # writing header
        f.write('# vtk DataFile Version 3.0\n')
        f.write('saved from python using save_gray2vtk\n')
        f.write('{}\n'.format(filetype))
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write('DIMENSIONS {} {} {}\n'.format(\
                volume.shape[2],volume.shape[1],volume.shape[0]))
        f.write('ORIGIN {} {} {}\n'.format(origin[0],origin[1],origin[2]))
        f.write('SPACING {} {} {}\n'.format(spacing[0],spacing[1],spacing[2]))
        f.write('POINT_DATA {}\n'.format(volume.size))
        f.write('SCALARS {} float 1\n'.format(dataname))
        f.write('LOOKUP_TABLE default\n')
        
    # writing volume data
    if filetype.upper()=='BINARY':
        with open(filename, 'ab') as f:
            volume = volume.astype('float32') # Pareview expects 4-bytes float 
            volume.byteswap(True) # Paraview expects big-endian 
            volume.tofile(f)
    else: # ASCII
        with open(filename, 'a') as f:
            np.savetxt(f,volume.ravel(),fmt='%.5g', newline= ' ')
        
def save_rgba2vtk(rgba, dim, filename, filetype='ASCII'):
    ''' Writes a vtk file with RGBA volume data.
    Arguments:
       rgba: an array of shape (N,4) containing RGBA values
       dim: volume shape, such that prod(dim) = N
       filename: filename with .vtk extension
       filetype: file type 'ASCII' or 'BINARY'. Writing a binary file might not
           work for all OS due to big/little endian issues.
    Author:vand@dtu.dk, 2019
    '''    
    with open(filename, 'w') as f:
        # writing header
        f.write('# vtk DataFile Version 3.0\n')
        f.write('saved from python using save_rgba2vtk\n')
        f.write('{}\n'.format(filetype))
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write('DIMENSIONS {} {} {}\n'.format(dim[2],dim[1],dim[0]))
        f.write('ORIGIN 0 0 0\n')
        f.write('SPACING 1 1 1\n')
        f.write('POINT_DATA {}\n'.format(np.prod(dim)))
        f.write('COLOR_SCALARS rgba 4\n')
    
    # writing color data
    if filetype.upper()=='BINARY':
        with open(filename, 'ab') as f:
            rgba = (255*rgba).astype('ubyte') # Pareview expects unsigned char  
            rgba.byteswap(True) # Paraview expects big-endian 
            rgba.tofile(f)
    else: # ASCII
        with open(filename, 'a') as f:
            np.savetxt(f,rgba.ravel(),fmt='%.5g', newline= ' ')   


def save_thickness2vtk(thickness, filename, **options):
    ''' Writes a vtk file with results of local thickness analysis.
    Author:vand@dtu.dk, 2019
    '''

    filetype = options.get('filetype', 'ASCII')
    origin = options.get('origin', (0, 0, 0))
    spacing = options.get('spacing', (1, 1, 1))
    
    thickness, gray, maxval = prepare_for_saving(thickness, options)

    cmap = black_plasma(bg=0, bg_alpha=0.1)  
    rgba = cmap(thickness.ravel()/maxval)

    # Now saving
    save_gray2vtk(gray, filename, filetype=filetype, origin=origin, spacing=spacing)
       
    with open(filename, 'a') as f:
        f.write('COLOR_SCALARS rgba 4\n')
    
    # writing color data
    if filetype.upper()=='BINARY':
        with open(filename, 'ab') as f:
            rgba = (255*rgba).astype('ubyte') # Pareview expects unsigned char  
            rgba.byteswap(True) # Paraview expects big-endian 
            rgba.tofile(f)
    else: # ASCII
        with open(filename, 'a') as f:
            np.savetxt(f,rgba.ravel(),fmt='%.5g', newline= ' ')  
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.filters import gaussian_laplace

def clean_frame(frame, median_radius=5, log_sigma=4):
    """
      Input: ndarray image, filter kernel settings
      Output: cleaned ndarray image
      A median filter is used to remove speckle noise, 
        followed by edge sharpening with a Laplacian 
        of Gaussian (LoG) mask.
    """

    # TODO scale input image in range (0,1)
    # TODO provide default for median_radius that is 
    #   sensitive to image dimensions

    frame = frame.astype(np.int64)
    medfilt = median_filter(frame, median_radius)
    logmask = gaussian_laplace(medfilt, log_sigma)
    cleaned = medfilt + logmask
    cleaned = cleaned.astype(np.uint8)
    
    return cleaned
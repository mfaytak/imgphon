'''
Filters for preprocessing lingual ultrasound data for dimensionality reduction.
  TODO: include generalized dimensionality reduction utilities here.
'''

import numpy as np

from scipy.ndimage import median_filter
from scipy.ndimage.filters import gaussian_laplace

def normalize(frame):
    """
    Normalize input image to range 0,1 and cast to float.
    """
    mx = float(np.amax(frame))
    mn = float(np.amin(frame))
    norm = (frame-mn)/(mx-mn)

    return norm

def norm_check(frame):
    """
    Check if a frame consists of floats normalized on 0,1.
    """
    if not np.issubdtype(frame.dtype, np.floating):
        raise TypeError("Input data must be float arrays")

    if not (frame >= 0.).all() and (frame <= 1.).all():
        raise ValueError("Input data must be normalized to range 0,1")


def srad(frame, n_iter=300, lbda=0.05):
    '''
    Speckle-reducing anisotropic diffusion filter to reduce noise
      typical of ultrasound images. Derived from MATLAB code in  
      Chris Carignan's TRACTUS repo 
      (https://github.com/ChristopherCarignan/TRACTUS/, in SRAD.m)
      which is in turn derived from the original algorithm in 
      Yu, Y. & Acton, S. (2002), "Speckle Reducing Anisotropic 
      Diffusion", IEEE Transactions on Image Processing 11(11), 
      DOI 10.1109/TIP.2002.804276.

    Inputs: frame, an ultrasound frame
      n_iter: number of iterations (Y&A use 300)
      lbda: lambda, AKA delta-t in Y&A (who use 0.05)
    Outputs: J, filtered ultrasound frame.
    '''

    # checks on I for number/type
    # TODO

    # scale to [0,1]
    I = normalize(frame)

    # get image size
    M,N = I.shape

    # image indices, using boundary conditions 
    iN = np.concatenate((np.arange(0, 1), np.arange(0, M-1)), axis=0)
    iS = np.concatenate((np.arange(1, M), np.arange(M-1, M)), axis=0) 
    jW = np.concatenate((np.arange(0, 1), np.arange(0, N-1)), axis=0)
    jE = np.concatenate((np.arange(1, N), np.arange(N-1, N)), axis=0)

    # log uncompress
    I = np.exp(I)

    # the algorithm itself
    for n in range(0,n_iter):

        # speckle scale fcn
        # IC = I.copy()
        # Iuniform = IC.crop(rect)
        q0_squared = np.var(I) / (np.mean(I)**2)

        # differences, element-by-element along each row moving from given direction (N, S, E, W)
        dN = I[iN,:] - I
        dS = I[iS,:] - I
        dW = I[:,jW] - I
        dE = I[:,jE] - I

        # normalized discrete gradient magnitude squared (Yu and Acton eqn. 52, 53)
        G2 = (dN**2 + dS**2 + dW**2 + dE**2) / I**2

        # normalized discrete Laplacian (eqn. 54)
        L = (dN + dS + dW + dE) / I

        # instantaneous coefficient of variation (ICOV) (eqns. 31/35)
        num = (.5*G2) - ((1/16)*(L**2))
        den = (1. + ((.25)*L))**2
        q_squared = num / (den + np.spacing(1))

        # diffusion coefficient (eqn. 33) # TODO why is this also "den"?
        den = (q_squared - q0_squared) / (q0_squared * (q0_squared + 1) + np.spacing(1))
        c = 1 / (den + 1)

        # saturate diffusion coefficient 
        c = np.where(c>0, 1, 0)

        # divergence (eqn. 58)
        cS = c[iS,:] 
        cE = c[:,jE] 
        D = (c * dN) + (cS * dS) + (c * dW) + (cE * dE)

        # SRAD update fcn (eqn. 61)
        I = I + (lbda/4) * D

    # log (re)compress
    J = np.log(I) 

    return J

def clean_frame(frame, median_radius=6, log_sigma=4):
    """
    Cleanup function to be run on SRAD output. Median filter for
      further denoising, followed by edge sharpening with a Laplacian 
      of Gaussian (LoG) mask.

    Inputs: ndarray image, filter kernel settings
      median_radius: median filter radius; should be odd integer
      log_sigma: LoG sigma; controls kernel size
    Output: cleaned; a processed ndarray
      
    """

    # TODO provide default for median_radius that is 
    #   sensitive to image dimensions

    norm_check(frame)

    # median filter
    cleaned = median_filter(frame, median_radius)

    # add LoG, protecting against overflow
    logmask = gaussian_laplace(cleaned, log_sigma)
    frame_ceil = np.finfo(frame.dtype).max
    logmask = frame_ceil - logmask
    np.putmask(cleaned, logmask < cleaned, logmask)
    cleaned += frame_ceil - logmask
    
    return cleaned

def noise_mask(frame):
    """
    TODO - expects normed, but before SRAD
    Adds random noise to an image. Possible processing step
      to be carried out before SRAD.
    Inputs:
      frame - ultrasound image
    Outputs:
      noised - ultrasound image with added random noise
    """
    
    norm_check(frame)

    noisemask = np.random.random_sample(0, 1, size=frame.shape)
    noised = frame + noise_mask # TODO truncate so no > 1


    return noised

def roi(frame, lower, upper, left, right):
    """
    Defines region of interest along ultrasound scan lines; returns
      boolean array in which 1 indicates an area inside the RoI
      and 0 outside the RoI. Can be multiplied with frame to mask.

    Inputs: 
      frame: ultrasound data in ndarray
      lower: bound of RoI further away from probe 
      upper: bound of RoI closer to probe
      left:

    Outputs:
      mask: ndarray of same shape as frame containing mask

    """

    if lower >= upper:
        raise ValueError("ROI lower bound must be smaller than upper bound")
    if left >= right:
        raise ValueError("ROI left bound must be smaller than right bound")

    mask = np.zeros(frame.shape, dtype=frame.dtype)
    mask[lower:upper,left:right] = 1

    return mask

def reconstruct_frame(vectors, values, num_components, image_shape, rescale=1):
    '''
    Access eigenvalues (from transformed data) and eigenvectors (from PCA) to reconstruct basis data
    Assuming a sklearn.decomposition.PCA object called "pca" and some basis data, inputs are:
      vectors: Eigenvectors, from pca.components_
      values: Eigenvalues for a token in basis data, AKA a first-level element in output of pca.transform(basisdata). 
              Can also use on subsets (multiple tokens) of data, in which case mean of each eigenvalue is used.
      num_components: Number of PCs, from pca.n_components.
      image_shape: The height and width of the images in the basis data (i.e., a tuple from basisdata[0].shape).
              Determines the dimensions of the "eigentongues" used in reconstruction.
      rescale: defaults to 1, for no rescaling. Multiply basis data by this scalar factor for
              display purposes.
    '''
    # for a given number of eigenvectors
    # multiply each by its eigenvalues
    if values.ndim > 1:
        rec_values = np.mean(values, axis=0)
    else:
        rec_values = values
        
    recon = None
    
    for pc in range(num_components):
        if recon is None:
            recon = vectors[pc].reshape(image_shape) * rec_values[pc]
        else:
            recon += vectors[pc].reshape(image_shape) * rec_values[pc]
            
    return rescale * recon

# TODO: group frames into training/test from a PD DataFrame

# TODO: PCA on arrays in short dimension (ideally, on frame bundles) - linked DataFrame?

# TODO: LDA on arrays - but what kind of object? DataFrame?

import numpy as np
import cv2
import bm3d

def bm3d_pypi(image, noise):
    denoised_image = bm3d.bm3d(image, noise, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return denoised_image

def bilateral_filter(image, sigma_d, sigma_r):
    image32f = image.astype(np.float32)
    denoised_image = cv2.bilateralFilter(
        image32f, d=int(3 * sigma_d), sigmaColor=sigma_r, sigmaSpace=sigma_d
    )
    return denoised_image

def anisodiff(image, kappa, gamma):
    n_iter = 2  # number of iterations
    step = (
        1.0,
        1.0,
    )  # distance between adjacent pixels in (y,x), used to scale if pixel spacing is different in the two directions

    # initializing input and output arrays
    image = image.astype(np.float32)
    denoised_image = image.copy()

    # initializing internal variables
    # differences
    deltaS = np.zeros_like(denoised_image)
    deltaE = deltaS.copy()
    # gradients
    gS = np.ones_like(denoised_image)
    gE = gS.copy()
    # gradient updates
    NS = deltaS.copy()
    EW = deltaS.copy()

    for i in range(n_iter):
        # calculating the neighbour differences
        deltaS[:-1, :] = np.diff(denoised_image, axis=0)
        deltaE[:, :-1] = np.diff(denoised_image, axis=1)

        # calculating the conduction gradients
        gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[0]
        gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[1]

        # updating the matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a (1 pixel shifted) copy of the matric
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        denoised_image += gamma * (NS + EW)

    return denoised_image
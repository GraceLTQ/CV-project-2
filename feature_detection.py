import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations

## Helper functions ############################################################


def inbounds(shape, indices):
    '''
        Input:
            shape -- int tuple containing the shape of the array
            indices -- int list containing the indices we are trying 
                       to access within the array
        Output:
            True/False, depending on whether the indices are within the bounds of 
            the array with the given shape
    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Compute Harris Values ############################################################
def computeHarrisValues(srcImage):
    '''
    Input:
        srcImage -- Grayscale input image in a numpy array with
                    values in [0, 1]. The dimensions are (rows, cols).
    Output:
        harrisImage -- numpy array containing the Harris score at
                        each pixel.
        orientationImage -- numpy array containing the orientation of the
                            gradient at each pixel in degrees.
    '''
    height, width = srcImage.shape[:2]

    harrisImage = np.zeros(srcImage.shape[:2])
    orientationImage = np.zeros(srcImage.shape[:2])

    # TODO 1: Compute the harris corner strength for 'srcImage' at
    # each pixel and store in 'harrisImage'. Also compute an
    # orientation for each pixel and store it in 'orientationImage.'

    # TODO-BLOCK-BEGIN

    # compute gradient using Sobel
    dx = ndimage.sobel(srcImage, 1)
    dy = ndimage.sobel(srcImage, 0)

    dx_sq = ndimage.gaussian_filter(dx**2, sigma=0.5, mode='nearest', radius=2)
    dx_dy = ndimage.gaussian_filter(dx*dy, sigma=0.5, mode='nearest', radius=2)
    dy_sq = ndimage.gaussian_filter(dy**2, sigma=0.5, mode='nearest', radius=2)

    orientationImage = np.arctan2(dy, dx)*180/np.pi

    for y in range(height):
        for x in range(width):
            h = np.zeros((2, 2))
            h[0, 0] = dx_sq[y, x]
            h[0, 1] = dx_dy[y, x]
            h[1, 0] = h[0, 1]
            h[1, 1] = dy_sq[y, x]
            harrisImage[y, x] = np.linalg.det(h) - 0.1 * np.trace(h)**2

    # TODO-BLOCK-END

    return harrisImage, orientationImage


## Compute Corners From Harris Values ############################################################


def computeLocalMaximaHelper(harrisImage):
    '''
    Input:
        harrisImage -- numpy array containing the Harris score at
                       each pixel.
    Output:
        destImage -- numpy array containing True/False at
                     each pixel, depending on whether
                     the pixel value is the local maxima in
                     its 7x7 neighborhood.
    '''
    destImage = np.zeros_like(harrisImage, dtype=bool)

    # TODO 2: Compute the local maxima image
    # TODO-BLOCK-BEGIN

    # Define the size of the neighborhood for local maxima detection
    neighborhood_size = 7
    # Apply the maximum filter with the specified neighborhood size
    max_filter = ndimage.maximum_filter(harrisImage, size=neighborhood_size)
    # Compare the Harris image to the maximum filter output to determine local maxima
    destImage = harrisImage == max_filter

    # TODO-BLOCK-END

    return destImage


def detectCorners(harrisImage, orientationImage):
    '''
    Input:
        harrisImage -- numpy array containing the Harris score at
                       each pixel.
        orientationImage -- numpy array containing the orientation of the
                            gradient at each pixel in degrees.
    Output:
        features -- list of all detected features. Entries should 
        take the following form:
        (x-coord, y-coord, angle of gradient, the detector response)

        x-coord: x coordinate in the image
        y-coord: y coordinate in the image
        angle of the gradient: angle of the gradient in degrees
        the detector response: the Harris score of the Harris detector at this point
    '''
    height, width = harrisImage.shape[:2]
    features = []

    # TODO 3: Select the strongest keypoints in a 7 x 7 area, according to
    # the corner strength function. Once local maxima are identified then
    # construct the corresponding corner tuple of each local maxima.
    # Return features, a list of all such features.
    # TODO-BLOCK-BEGIN

    destImage = computeLocalMaximaHelper(harrisImage)
    for y in range(height):
        for x in range(width):
            if destImage[y, x]:
                features.append(
                    (x, y, orientationImage[y, x], harrisImage[y, x]))

    # TODO-BLOCK-END

    return features


## Compute MOPS Descriptors ############################################################
def computeMOPSDescriptors(image, features):
    """"
    Input:
        image -- Grayscale input image in a numpy array with
                values in [0, 1]. The dimensions are (rows, cols).
        features -- the detected features, we have to compute the feature
                    descriptors at the specified coordinates
    Output:
        desc -- K x W^2 numpy array, where K is the number of features
                and W is the window size
    """
    image = image.astype(np.float32)
    image /= 255.
    # This image represents the window around the feature you need to
    # compute to store as the feature descriptor (row-major)
    windowSize = 8
    desc = np.zeros((len(features), windowSize * windowSize))
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)

    for i, f in enumerate(features):
        transMx = np.zeros((2, 3))

        # TODO 4: Compute the transform as described by the feature
        # location/orientation and store in 'transMx.' You will need
        # to compute the transform from each pixel in the 40x40 rotated
        # window surrounding the feature to the appropriate pixels in
        # the 8x8 feature descriptor image. 'transformations.py' has
        # helper functions that might be useful
        # Note: use grayImage to compute features on, not the input image
        # TODO-BLOCK-BEGIN
        x, y, angle, _ = f
        # Convert angle to radians and negate for clockwise rotation
        angle = np.radians(-angle)

        # Adjustments for 3D transformation functions
        trans_vec = np.array([-x, -y, 0])
        # Scale factors with unity z-component to ignore depth scaling
        s_x, s_y, s_z = 0.2, 0.2, 1

        # Compute 3D transformation matrices
        T1 = transformations.get_trans_mx(trans_vec)
        R = transformations.get_rot_mx(0, 0, angle)
        S = transformations.get_scale_mx(s_x, s_y, s_z)
        T2 = transformations.get_trans_mx(
            np.array([windowSize / 2, windowSize / 2, 0]))

        # Combine transformations into a single matrix and extract the upper-left 2x3 matrix for 2D affine transformation
        transMx_3D = T2 @ S @ R @ T1
        transMx_3D[:, 2] = transMx_3D[:, 3]
        transMx = transMx_3D[:2, :3]  # Extract 2x3 matrix for cv2.warpAffine

        # TODO-BLOCK-END

        destImage = cv2.warpAffine(grayImage, transMx,
                                   (windowSize, windowSize), flags=cv2.INTER_LINEAR)

        # TODO 5: Normalize the descriptor to have zero mean and unit
        # variance. If the variance is negligibly small (which we
        # define as less than 1e-10) then set the descriptor
        # vector to zero. Lastly, write the vector to desc.
        # TODO-BLOCK-BEGIN
        # Normalize the descriptor
        mean = np.mean(destImage)
        std = np.std(destImage)
        if std ** 2 < 1e-10:
            desc[i, :] = 0  # Avoid division by zero or near zero variance
        else:
            normalized = (destImage - mean) / std
            # Store the normalized descriptor
            desc[i, :] = normalized.flatten()
        # TODO-BLOCK-END

    return desc


## Compute Matches ############################################################
def produceMatches(desc_img1, desc_img2):
    """
    Input:
        desc_img1 -- corresponding set of MOPS descriptors for image 1
        desc_img2 -- corresponding set of MOPS descriptors for image 2

    Output:
        matches -- list of all matches. Entries should 
        take the following form:
        (index_img1, index_img2, score)

        index_img1: the index in corners_img1 and desc_img1 that is being matched
        index_img2: the index in corners_img2 and desc_img2 that is being matched
        score: the scalar difference between the points as defined
                    via the ratio test
    """
    matches = []
    assert desc_img1.ndim == 2
    assert desc_img2.ndim == 2
    assert desc_img1.shape[1] == desc_img2.shape[1]

    if desc_img1.shape[0] == 0 or desc_img2.shape[0] == 0:
        return []

    # TODO 6: Perform ratio feature matching.
    # This uses the ratio of the SSD distance of the two best matches
    # and matches a feature in the first image with the closest feature in the
    # second image. If the SSD distance is negligibly small, in this case less
    # than 1e-5, then set the distance to 1. If there are less than two features,
    # set the distance to 0.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # TODO-BLOCK-BEGIN

    # Calculate the pairwise distances between descriptors
    dist_matrix = spatial.distance.cdist(desc_img1, desc_img2, 'euclidean')
    dist_matrix = dist_matrix**2


    dist_matrix[dist_matrix < 1e-5] = 1
    if (len(desc_img1) < 2 or len(desc_img2) < 2):
        dist_matrix = np.zeros((desc_img1.shape[0],desc_img2.shape[0]))
    

    # Iterate over each descriptor in image 1
    for idx1, distances in enumerate(dist_matrix):           

        # Get the indices of the sorted distances (ascending)
        sorted_indices = np.argsort(distances)

        # Get the smallest and second smallest distance
        closest, second_closest = sorted_indices[0], sorted_indices[1]
        closest_distance, second_closest_distance = distances[closest], distances[second_closest]

        # The score can be defined as the ratio of the distances
        score = closest_distance / second_closest_distance
        matches.append((idx1, closest, score))
    # TODO-BLOCK-END

    return matches

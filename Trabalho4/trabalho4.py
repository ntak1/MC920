# coding: utf-8

import numpy as np
import cv2
import os
import random
import time
from utils import *

print('OpenCV version:', cv2.__version__)

# Load images
image_dir = 'imagens_registro/'
image_names = sorted(os.listdir(image_dir))

images = [cv2.imread(os.path.join(image_dir,name)) for name in image_names]

# Convert to gray scale
images_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

# Define de algorithms to be used
algorithms = ['sift', 'surf', 'orb','brief']

# Create the output directories
for algorithm in algorithms:
    dir_name = os.path.join('output', algorithm)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
for algorithm in algorithms:
    
    save_name = algorithm
    
    # Construct the detector object
    if algorithm == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif algorithm == 'surf':
        detector = cv2.xfeatures2d.SURF_create()
        norm = cv2.NORM_L2
    elif algorithm == 'orb':
        norm = cv2.NORM_HAMMING
        detector = cv2.ORB_create()
    elif algorithm == 'brief':
        # Initiate FAST detector
        norm = cv2.NORM_HAMMING
        detector = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        
    for pair_index in range(len(images_gray)//2):
        print(algorithm, 'pair_index = ', pair_index)
        # Detect key points
        start = time.time()
        key_points1 = detector.detect(images_gray[2*pair_index],None)
        img1=cv2.drawKeypoints(images_gray[2*pair_index],key_points1, images_gray[2*pair_index],
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        key_points2 = detector.detect(images_gray[2*pair_index+1],None)
        img2=cv2.drawKeypoints(images_gray[2*pair_index+1],key_points2, images_gray[2*pair_index+1],
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the draw
        cv2.imwrite(os.path.join('output', algorithm, algorithm + str(pair_index) + '_kp1.jpg'), img1)
        cv2.imwrite(os.path.join('output', algorithm, algorithm + str(pair_index) + '_kp2.jpg'), img2)
        
        # Compute descritors
        if not algorithm == 'brief':
            _, des1 = detector.compute(images_gray[2*pair_index],key_points1)
            _, des2 = detector.compute(images_gray[2*pair_index+1],key_points2)
        else:
            # compute the descriptors with BRIEF
            key_points1, des1 = brief.compute(images_gray[2*pair_index], key_points1)
            key_points2, des2 = brief.compute(images_gray[2*pair_index + 1], key_points2)
            
        # create BFMatcher (brute force) object
        bf = cv2.BFMatcher(norm, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(des1,des2)


        # Sort them in the order of their distance. (The lower the better)
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw matches
        img3 = cv2.drawMatches(images_gray[2*pair_index],key_points1,images_gray[2*pair_index+1],
                              key_points2,matches, flags=2,outImg=images_gray[0])
#         plt.imshow(img3),plt.show()
        cv2.imwrite(os.path.join('output', algorithm, algorithm + str(pair_index) + '_match.jpg'), img3)

        # Extract points
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for index, match in enumerate(matches):
            points1[index,:] = key_points1[match.queryIdx].pt # source
            points2[index,:] = key_points2[match.trainIdx].pt # destiny

 
        # Find homography using RANSAC
        assert(points1 is not None and points2 is not None)
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)
        print('num_matches = ', len(matches))
        img_right = images[2*pair_index] # Right
        img_left= images[2*pair_index+1] # Left

        dst = cv2.warpPerspective(img_right,h,(img_left.shape[1] + img_right.shape[1], img_left.shape[0]))
        dst[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        
        end = time.time()
        print('time elapsed = ',  end - start, '\n')
        cv2.imwrite(os.path.join('output', algorithm, algorithm + str(pair_index)) + '.jpg',dst)
        
        
        





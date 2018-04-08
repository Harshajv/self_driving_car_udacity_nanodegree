import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
import collections

def undistort(img,mtx,dist,verbose= False):
    ##calculate the distortion coffecient and calibration matrix use cv2.calibrateCamera
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    ##undistort input images
    undist=cv2.undistort(img, mtx, dist, None, mtx)
    
    if verbose:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    
    
    return undist


def cam_calibration(image_dir,nx,ny,verbose=False):
    
    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    # Step through the list and search for chessboard corners
    
    
    # make a list of calibration images
    images= glob.glob(os.path.join(image_dir,"calibration*.jpg"))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        if verbose:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #cv2.imshow('img',img)
            plt.imshow(img)
            save_doc_img("find_corners",img)
#cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret,mtx,dist,rvecs,tvecs


if __name__ == '__main__':
    ret,mtx,dist,rvecs,tvecs= cam_calibration(image_dir="../camera_cal/",nx=9,ny=6)
    test_image= mpimg.imread("../camera_cal/calibration1.jpg")
    #print("image before undistort")
    #plt.imshow(test_image)
    #undistorted_test_image=undistort(test_image,mtx,dist)
    undistorted_test_image=undistort(test_image,mtx,dist,verbose= True)
    cv2.imwrite('/Users/likangning/CarND-Advanced-Lane-lines/output_images/test_calibration_before.jpg', test_image)
    cv2.imwrite('/Users/likangning/CarND-Advanced-Lane-lines/output_images/test_calibration_after.jpg', undistorted_test_image)




    


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration
import binarization_mask_calculation


def perspect_trans(img,verbose=False):
    ####apply perspective transform to get the birdview of image
    img_size=(img.shape[1],img.shape[0])
    x1=205
    x2=1100
    y_rough=460
    offset= 100
    
    
    src=((x1,img.shape[0]),(x2,img.shape[0]),(700,y_rough),(585,y_rough))
    
    dst=((x1+offset,img.shape[0]),(x2-offset,img.shape[0]),(x2-offset,0),(x1+offset,0))
    
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    plt.imshow(warped)
    
    
    if verbose:
        #image_visual=img.copy()
        for i in range(4):
            img=cv2.line(img,src[i],src[(i+1)%4], color=[255,0,0],thickness=4)
            warped=cv2.line(warped,dst[i],dst[(i+1)%4], color=[255,0,0],thickness=4)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(warped,cmap="gray")
        ax2.set_title('BirdEye Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    return M,Minv,warped


if __name__ == '__main__':

   print("done")



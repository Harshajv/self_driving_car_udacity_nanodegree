import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', thresh=[0,255]):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=="x":
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    elif orient=="y":
        sobelx= cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude
# is > thresh_min and < thresh_max
    binary_output=np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    ##Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread().
    ##Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
    # 1) Convert to grayscale
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude
    abs_sobelxy=np.sqrt(sobelx**2+sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    abs_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_output=np.zeros_like(abs_sobelxy)
    binary_output[(abs_sobelxy >= mag_thresh[0]) & (abs_sobelxy <= mag_thresh[1])] = 1
    # 6) Return this mask as binary_output image
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray_img= cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx=cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    sobely=cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the x and y gradients
    sobelx=np.absolute(sobelx)
    sobely=np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad=np.arctan2(sobely,sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output=np.zeros_like(grad)
    binary_output[(grad >= thresh[0]) & (grad <= thresh[1])] = 1
    return binary_output

def compute_sobel_thresh(image,ksize,verbose=False):
    gradx = abs_sobel_thresh(image, orient='x', thresh=(25, 255))
    grady = abs_sobel_thresh(image, orient='y', thresh=(25, 255))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 2.0))
    
    if verbose:
        f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(gradx,cmap="gray")
        ax1.set_title('Gradx', fontsize=50)
        ax2.imshow(grady,cmap="gray")
        ax2.set_title('Grady', fontsize=50)
        ax3.imshow(mag_binary,cmap="gray")
        ax3.set_title('mag_binary', fontsize=50)
        ax4.imshow(dir_binary,cmap="gray")
        ax4.set_title('dir_binary', fontsize=50)
        
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    return gradx, grady, mag_binary,dir_binary
def extract_yellow(image,thresh,verbose=False):##[170,255]
    ### s_channel of HLS is the most important for yellow extraction.
    hls_image=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    s_channel=hls_image[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    if verbose:
        plt.figure()
        plt.imshow(s_binary,cmap="gray")
        plt.title("extract_yellow_binary")
    
    return s_binary

#yellow_lower = np.array([15,50,170])
#yellow_upper = np.array([25,200,255])
#return cv2.inRange(img_hls, yellow_lower, yellow_upper) // 255
def extract_yellow_update(image,verbose=False):
    hls_image=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    yellow_lower = np.array([15,50,100])
    yellow_upper = np.array([25,200,255])
    
    extract_yellow_binary=cv2.inRange(hls_image, yellow_lower, yellow_upper) // 255
    if verbose:
        plt.figure()
        plt.imshow(extract_yellow_binary,cmap="gray")
        plt.title("extract_yellow_binary")
    
    return extract_yellow_binary

def extract_white(image,verbose=False):
    hls_image=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    white_lower = np.array([0,  200, 0])
    white_upper = np.array([255,255, 255])
    
    extract_white_binary=cv2.inRange(hls_image, white_lower, white_upper) // 255
    if verbose:
        plt.figure()
        plt.imshow(extract_white_binary,cmap="gray")
        plt.title("extract_white_binary")
    
    return extract_white_binary


def combined_color_mask(image,verbose=False):
    yellow_binary=extract_yellow_update(image)
    white_binary=extract_white(image)
    color_mask=cv2.bitwise_or(yellow_binary, white_binary)
    
    if verbose:
        plt.imshow(color_mask,cmap="gray")
        plt.title("combined_color_mask")
    
    return color_mask


def mask_to_rgb(img):
    return 255 * np.dstack((img, img, img))


def final_mask_combination(img,verbose=False):###yellow_thresh[170,255]
    #yellow_binary=extract_yellow_update(img,verbose= False)
    #white_binary=extract_white(img,verbose= False)
    color_mask = combined_color_mask(img)
    
    ##then compute the sobel gradient thresh.
    gradx,grady,mag_binary,dir_binary=compute_sobel_thresh(img,ksize=3,verbose=False)
    #combined_sobel_mask=combined_sobel_gradient(gradx,mag_binary)
    Final_mask=cv2.bitwise_or(color_mask,gradx)
    #Final_mask=mask_to_rgb(Final_mask)
    #Final_mask= cv2.bitwise_or(color_mask,combined_sobel_mask)
    
    if verbose:
        plt.imshow(Final_mask,cmap="gray")
    return Final_mask






import cv2
import sys
from BayesianMatting import BayesianMatting
from ShowNwriteIm import ShowNwriteIm
from trimap_module import trimap, remove_green
import numpy as np


if __name__ == '__main__':

  def composite_bg(img, bg, alpha, scale_percent = 100):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize
    img = cv2.resize(img, dim)
    alpha = cv2.resize(alpha, dim)

    x_ini = int((bg.shape[1]-img.shape[1])/2)
    x_end = int((bg.shape[1]+img.shape[1])/2)
    #x_ini = int(-img.shape[1])
    #x_end = int(bg.shape[1])
    y_ini = -img.shape[0]
    y_end = bg.shape[0]
    
    bg_crop = bg[y_ini:y_end, x_ini:x_end, :]
    bg_crop = alpha*img + (1-alpha)*bg_crop
    new_image = bg.copy()
    new_image[y_ini:y_end, x_ini:x_end, :]  = bg_crop

    return new_image

  #User input
  indexImage = input("Please input the index of the Original Image: ")
  indexBackground = input("Please input the index of the Background Image: ")

  #set parameters
  kernel_size = 31  # initial kernel size (odd)
  min_pix = 150  # minimum required pixels in a kernel
  sigma = 8  # variance of Gaussian mask
  threshold = 1e-5  # stopping threshold
  max_it = 150  # maximum iterations
  
  # Read images
  Original_Image = cv2.imread("inputs/Original_Image/image_"+str(indexImage)+".png")[:, :, :3]

  # Remove green
  Ground_Truth = remove_green(Original_Image)
  Ground_Truth = np.repeat(Ground_Truth[:, :, np.newaxis], 3, axis=2)
  cv2.imwrite("./inputs/Ground_Truth/gt_"+str(indexImage)+".png", Ground_Truth)

  # Create trimap
  Trimap = trimap(Ground_Truth[:,:,0], size = 25, erosion=6)
  Trimap = np.repeat(Trimap[:, :, np.newaxis], 3, axis=2)
  cv2.imwrite("./inputs/Trimap/trimap_"+str(indexImage)+".png", Trimap)

  # Color sengmentation
  Background = cv2.imread("./inputs/Background/background_"+indexBackground+".png")
  removed = composite_bg(Original_Image, Background, Ground_Truth/255, scale_percent = 100)
  cv2.imwrite("./outputs/green_removed_"+str(indexImage)+".png", removed)

  #Bayesian Matting
  new_alpha_map, diff, final_im, im_diff, new_fore, new_image = \
  BayesianMatting(kernel_size, min_pix, sigma, threshold, max_it, Ground_Truth,
                  Original_Image, Trimap, Background)
  # Outputs
  ShowNwriteIm(new_alpha_map, diff, final_im, im_diff, new_fore, new_image, indexImage)








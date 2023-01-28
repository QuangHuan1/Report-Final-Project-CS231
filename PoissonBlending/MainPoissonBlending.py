import cv2
import sys
import getopt
import os
from os import path
from move_mask import MaskMover
from poisson_image_editing import poisson_edit


SourceScale = 90
TargetScale = 60

def resize(src, scale_percent):
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv2.resize(src, dsize, interpolation = cv2.INTER_AREA)
    return output

if __name__ == '__main__':
    # #    
    IMAGES_FOLDER_PATH = "./image"
    sourceImagePath = os.path.join(IMAGES_FOLDER_PATH, 'fg/estimated_image_5.png')
    targetImagePath = os.path.join(IMAGES_FOLDER_PATH, 'bg/bg1.png')
    maskImagePath = os.path.join(IMAGES_FOLDER_PATH, 'mask/new_alpha_map_5.png')

    source = cv2.imread(sourceImagePath)
    target = cv2.imread(targetImagePath)
    # source = cv2.imread(args["source"])
    # target = cv2.imread(args["target"])
    
    source = resize(source, SourceScale)
    target = resize(target, TargetScale)
   
    # adjust mask position for target image
    print('\nPlease move the object to desired location to apparate.\n')
    mm = MaskMover(targetImagePath, maskImagePath)
    offset_x, offset_y, target_mask_path = mm.move_mask()            

    # blend
    print('Blending ...')
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE) 
    offset = offset_x, offset_y

    poisson_blend_result = poisson_edit(source, target, target_mask, offset)
    
    cv2.imwrite(path.join("./image/blendingresult", 'newresult.png'), 
                poisson_blend_result)
    
    print('Done.\n')
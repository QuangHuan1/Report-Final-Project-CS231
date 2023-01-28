import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from trimap_module import remove_green, trimap
from utils import *


IMAGES_FOLDER_PATH = "./image"
img3 = cv2.imread(os.path.join(IMAGES_FOLDER_PATH, 'fg/fg3.jpg'))
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img3_gray = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

new_bg3 = cv2.imread(os.path.join(IMAGES_FOLDER_PATH, 'bg/bg9.jpg'))
new_bg3 = cv2.cvtColor(new_bg3, cv2.COLOR_BGR2RGB)
new_bg3 = cv2.resize(new_bg3, (img3.shape[1],img3.shape[0])) 

mask = remove_green(img3)
tri3 = trimap(mask, size = 20, erosion=True)
savedname = os.path.join(IMAGES_FOLDER_PATH, 'trimap/trimapfg3.png')
plt.imsave(savedname, tri3, cmap=cm.gray)

display_img_arr([img3,img3_gray,tri3,new_bg3], 2,2,(10,10), ['original img','grayscale','tri-map','new-bg'])

print("Global Matting.....")
all_data = matting_combined(tri3, img3_gray)
all_data.update({'img3': img3, 'img3_gray': img3_gray})
## NOW all_data VARIABLE CONTAINS IMG3, IMG3_GRAY, ALPHA, DIFF, F, B, UNKNOWN
display_img_arr([all_data['alpha']], 1,1,(10,10),['alpha global matting'])

#alpha blending
new_img = alpha_blend(new_bg3,all_data['alpha'],img3)
display_img_arr([new_img], 1,1,(10,10),['Alpha blend w Global matte'])

#Local Matting
print("Local Matting.....")
local_matte = local_matting(all_data.copy())
display_img_arr([local_matte], 1,1,(10,10),['local_matte'])
new_img = alpha_blend(new_bg3,local_matte,img3)
display_img_arr([new_img ], 1,1,(10,10),['Alpha blend w Local matte'])





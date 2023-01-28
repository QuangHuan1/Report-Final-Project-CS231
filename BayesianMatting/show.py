import cv2

img = cv2.imread("./inputs/Original_Image/image_1.png")
bg = cv2.imread("./inputs/BackGround/background_2.png")
alpha = cv2.imread("./outputs/new_alpha_map.png", 0)

alpha = alpha/255
cv2.imshow("alpha",alpha)
def composite_bg(img, bg, alpha, scale_percent = 100):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize
    img = cv2.resize(img, dim)
    alpha = cv2.resize(alpha, dim)

    x_ini = int((bg.shape[1]-img.shape[1])/2)
    x_end = int((bg.shape[1]+img.shape[1])/2)
    y_ini = -img.shape[0]
    y_end = bg.shape[1]
    
    bg_crop = bg[y_ini:y_end, x_ini:x_end, :]
    bg_crop[:,:,0] = alpha*img[:,:,0] + (1-alpha)*bg_crop[:,:,0]
    bg_crop[:,:,1] = alpha*img[:,:,1] + (1-alpha)*bg_crop[:,:,1]
    bg_crop[:,:,2] = alpha*img[:,:,2] + (1-alpha)*bg_crop[:,:,2]
    bg[y_ini:y_end, x_ini:x_end, :]  = bg_crop

    return bg

result = composite_bg(img, bg, alpha, scale_percent = 100)

cv2.imwrite("outputs/composit_bg.png", result)

cv2.imshow("res", result)
cv2.waitKey(0)
import numpy as np
# from scipy.misc import imread, imresize
from PIL import Image
def cal_distance(samples, ground_th):
    
    x_s, y_s, w_s, h_s = np.array(samples, dtype='float')
    x_gt,y_gt,w_gt,h_gt = np.array(ground_th, dtype='float')

    deltaX = (x_gt + w_gt/2 - x_s - w_s/2) / w_s
    deltaY = (y_gt + h_gt/2 - y_s - h_s/2) / h_s
#    deltaX = (x_gt - x_s) / w_s
#    deltaY = (y_gt - y_s) / h_s
    deltaW = np.log(w_gt / w_s)
    deltaH = np.log(h_gt / h_s)
    
    return deltaX, deltaY, deltaW, deltaH

def move_bbox(pos_, deta_pos, img_size):

    deta_pos = np.squeeze(deta_pos)

#    # border value
#    deta_pos[0:2] = np.clip(deta_pos[0:2], -1.0, 1.0)
#    deta_pos[2:] = np.clip(deta_pos[2:], -0.05, 0.05)
    if abs(deta_pos[2]) > 0.08:
        deta_pos[2] = 0
    if abs(deta_pos[3]) > 0.08:
        deta_pos[3] = 0


    
    pos_deta = deta_pos[0:2] * pos_[2:]
    pos = np.copy(pos_)

    center = pos[0:2] + pos[2:4] / 2 + pos_deta
    pos[2:4] = pos[2:4] * np.exp(deta_pos[2:4])
    pos[0:2] = center - pos[2:4] / 2
#    pos[2:4] = pos[2:4] * np.exp(deta_pos[2:4])
#    pos[0:2] = pos[0:2] + pos_deta
    

    # the smallest value of w and h is set 10 
    pos[2] = max(pos[2], 10)
    pos[3] = max(pos[3], 10)
    pos[2] = min(pos[2], img_size[1]-1)
    pos[3] = min(pos[3], img_size[0]-1)    
     
    # To make sure the bbox overlapped with the valid image area.
    # Set the boundry to img_size-1, otherwise the crop_image func may collapse
    pos[0] = min(pos[0], img_size[1]-1)
    pos[1] = min(pos[1], img_size[0]-1)
    pos[0] = max(pos[0], -pos[2])
    pos[1] = max(pos[1], -pos[3])

    return pos     

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


def crop_image(img, bbox, img_size=107, padding=0, valid=False):
    x, y, w, h = bbox

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w = img.size
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w-1, max_x)
        max_y_val = min(img_h-1, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
                = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    scaled_l = cv2.resize(cropped, (img_size, img_size))
        
    min_x = int(center_x - w + 0.5)
    min_y = int(center_y - h + 0.5)
    max_x = int(center_x + w + 0.5)
    max_y = int(center_y + h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    scaled_g = cv2.resize(cropped, (img_size, img_size))

    return np.array([scaled_g, scaled_l])



def crop_resize(img, bbox, img_size=107, padding=0, valid=False):
    x, y, w, h = [int(round(num)) for num in bbox]
    #x, y, w, h = bbox
    
    
    img_h, img_w, _ = img.size
    x = max(0, x)
    y = max(0, y)
    xend = min(img_w, x+w)
    yend = min(img_h, y+h)
    
    cropped = img[y:yend, x:xend]
    
    scaled = cv2.resize(cropped, (img_size, img_size))

    return scaled

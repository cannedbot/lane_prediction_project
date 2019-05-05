'''
Predict and evaluate DNN based lane detection
'''

# TODO:
# 1. draw ground-truth mask with "r" radius
# 2. get prediction coordinates
# 3. output precision and recall

import cv2
import numpy as np
import math
import os

file_list = []
average_list = []

src_folder = './EVT1_20171127_002424/labeled'

coeL_test = ['-0.0009994328149954112', '-0.7179319813026466', '855.8099539091571']
coeR_test = ['0.0004090902442033915', '0.42832195778720594', '319.3596711459802']


def get_mask_coordinates(mask_image, channel='B', val=255):
    if channel == 'B':
        cval = 0
    elif channel == 'G':
        cval = 1
    elif channel == 'R':
        cval = 2

    coord = np.argwhere(mask_image[:, :, cval] == val)

    # print(coord)

    _y = coord[:, 0]
    _x = coord[:, 1]

    # print('x:', _x)
    # print('y:', _y)

    return _x, _y


def get_two_lane_labels(_line):
    data_raw = []
    coe1 = []
    coe2 = []

    _line = _line.strip('\r\n')
    _line += ' '
    line_content = _line.split(' ')

    for i in range(len(line_content) - 1):
        data_raw.append(line_content[i])

    for i in range(3):
        coe1.append(data_raw[i])

    for i in range(3, 6):
        coe2.append(data_raw[i])

    return coe1, coe2


def drawOneLane(_image, coe, color, h_hat, c_radius=1):
    colorDict = {"blue": (255, 0, 0),
                 "green": (0, 255, 0),
                 "yellow": (0, 255, 255),
                 "red": (0, 0, 255),
                 "white": (255, 255, 255)}

    xarray = [0, 0, 0]
    counter = 0

    for i in range(h_hat):
        a = float(coe[0])
        b = float(coe[1])
        c = float(coe[2]) - i
        # a = _a
        # b = _b
        # c = _c
        # print("a:", a, "b:", b, "c:", c)
        # print("_a:", _a, "_b:", _b, "_c:", _c - i)

        if a == 0:
            a = 0.0001

        if (b ** 2 - (4 * a * c)) < 0: continue

        d = math.sqrt((b ** 2) - (4 * a * c))
        x = (-b - d) // (2 * a)

        # else:
        #     x = 0

        # print('det: ', -b - d, 'x_hat:', x, ',y_hat:', i)
        cv2.circle(_image, (i, int(x)), c_radius, colorDict[color], thickness=-1, lineType=8, shift=0)
        x = (-b + d) // (2 * a)
        cv2.circle(_image, (i, int(x)), c_radius, colorDict[color], thickness=-1, lineType=8, shift=0)

    return _image


def evaluateLane(_image_mask, x_list, y_list, scan_limit, text):
    colorDict = {"blue": (255, 0, 0),
                 "green": (0, 255, 0),
                 "yellow": (0, 255, 255),
                 "red": (0, 0, 255),
                 "white": (255, 255, 255)}
    # for j in range(len(x_list)):
    #     print(x_list[j], y_list[j])
    #     cv2.circle(_image_mask, (x_list[j], y_list[j]), 4, (255, 0, 0), thickness=-1, lineType=8, shift=0)

    # TP - True Positive, FP - False Positive
    counter_TP = 0
    counter_FP = 0

    # print("Xmax:", max(x_list))
    # print("Ymax:", max(y_list))

    for i in range(len(x_list)):
        # print(x_list[i], y_list[i])
        # print("px: ", _image_mask[y_list[i], x_list[i]])
        # print("max: ", np.amax(_image_mask[y_list[i], x_list[i]]) == 255)
        # print(y_list[i], "<",scan_limit, "=", y_list[i] < scan_limit)
        if y_list[i] > scan_limit:
            # print("max:", _image_mask[y_list[i], x_list[i]])
            # print("mask:", _image_mask[0, 0])
            if np.amax(_image_mask[y_list[i], x_list[i]]) == 255:
                # print("--")
                # cv2.circle(_image_mask, (x_list[i], y_list[i]), 4, (255, 255, 0), thickness=-1, lineType=8, shift=0)
                counter_TP += 1
            else:
                # print("no")
                # cv2.circle(_image_mask, (x_list[i], y_list[i]), 4, (100, 100, 100), thickness=-1, lineType=8, shift=0)
                counter_FP += 1

    # cv2.line(_image_mask, (0, 270), (_image_mask.shape[1], 270), (255, 255, 255), 1)
    print(text + ":" + "TP:", counter_TP, "FP:", counter_FP)
    return counter_TP, counter_FP,


for filename in os.listdir(src_folder):
    if filename.endswith('lines.txt'): continue
    if filename.endswith('.jpg'): continue
    fullname = os.path.join(filename)
    basename = fullname.split('.')[0]
    file_list.append(basename)

for i in range(len(file_list)):
    with open(src_folder + '/' + file_list[i] + '.txt') as f:
        line = f.readline()
        print(line)

    coe1, coe2 = get_two_lane_labels(line)
    print("coe1:", coe1)
    print("coe2:", coe2)

    image = cv2.imread(src_folder + '/' + file_list[i] + '.jpg')
    img_h = image.shape[0]
    img_w = image.shape[1]
    # blank_image = image
    height_hat = img_w

    # mask for ground-truth (gt) and predicted (pd)
    mask_image_gt_left_lane = np.zeros([img_h, img_w, 3], np.uint8)
    mask_image_gt_right_lane = np.zeros([img_h, img_w, 3], np.uint8)

    mask_image_pd_left_lane = np.zeros([img_h, img_w, 3], np.uint8)
    mask_image_pd_right_lane = np.zeros([img_h, img_w, 3], np.uint8)

    drawOneLane(mask_image_gt_left_lane, coe1, "red", height_hat, c_radius=10)
    drawOneLane(mask_image_gt_right_lane, coe2, "green", height_hat, c_radius=10)

    drawOneLane(mask_image_pd_left_lane, coeL_test, "white", height_hat)
    # drawOneLane(mask_image_gt_left_lane, coeL_test, "white", height_hat)
    drawOneLane(mask_image_pd_right_lane, coeR_test, "white", height_hat)
    # drawOneLane(mask_image_gt_right_lane, coeR_test, "white", height_hat)

    # channel doesn't matter since pd line is white
    xl, yl = get_mask_coordinates(mask_image_pd_left_lane, channel='B', val=255)
    xr, yr = get_mask_coordinates(mask_image_pd_right_lane, channel='B', val=255)

    Ltp, Lfp = evaluateLane(mask_image_gt_left_lane, xl, yl, scan_limit=img_h / 2, text="left")
    Rtp, Rfp = evaluateLane(mask_image_gt_right_lane, xr, yr, scan_limit=img_h / 2, text="right")

    LeftPrecision = Ltp / (Ltp + Lfp)
    RightPrecision = Rtp / (Rtp + Rfp)

    average_precision = (LeftPrecision + RightPrecision) / 2

    average_list.append(average_precision)

    print(img_w / 2)
    # print("xl:", xl, "yl:", yl)
    # print("xl:", xr, "yl:", yr)

    # image = cv2.resize(image, None, fx=0.5, fy=0.5)
    # image = cv2.resize(image, (960, 540))
    print(image.shape)
    cv2.imshow("image", image)
    cv2.imshow("L-mask", mask_image_gt_left_lane)
    cv2.imshow("R-mask", mask_image_gt_right_lane)
    # cv2.imshow("pd-mask", mask_image_pd_left_lane)
    keysave = cv2.waitKey(1) & 0xFF

total_average = sum(average_list) / len(average_list)
print("total precision of folder:", total_average)

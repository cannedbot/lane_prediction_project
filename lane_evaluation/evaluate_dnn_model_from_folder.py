'''
Predict and evaluate DNN based lane detection
'''
import cv2
import numpy as np
import math
import os
import tensorflow as tf
import mobileNet_v2

file_list = []
average_list = []

# src_folder = './EVT1_20171127_002424/labeled'
src_folder = './culane_validation_dataset_with_coeficient_labels_x11x'

coeL_test = [-0.0009994328149954112, -0.7179319813026466, 855.8099539091571]
coeR_test = [0.0004090902442033915, 0.42832195778720594, 319.3596711459802]

default_label_type = False  # default is 6 labels False is culane 16 labels type
normalized_dataset = True
normalized_file_name = "./d_filtered_augmented_dataset/NormalizationParams.txt"
root_folder = "./f_augment_folder_v1_backup/"
checkpoint = root_folder + "LANE_finetune_Checkpoint/FineTuneCheckpoint"
output_regression_count = 6


def unnormalizeFromParams(output, norm_params):
    """from a normalized element it unnormalizes it back again so that we can interpret the result"""
    assert (len(norm_params) / len(output)) == 2, "There must be two norm params for every output!" + str(
        len(norm_params)) + " " + str(len(output))

    for i in range(len(output)):
        output[i] = (float(output[i]) * float(norm_params[i * 2 + 1])) + float(
            norm_params[i * 2])  # multiply by the std deviation and add the mean

    # output[0] = (float(output[0]) * float(norm_params[1])) + float(norm_params[0])
    # output[1] = (float(output[1]) * float(norm_params[3])) + float(norm_params[2])
    # output[2] = (float(output[2]) * float(norm_params[5])) + float(norm_params[4])

    return output


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


def get_two_lane_labels(_line, _labels_type):
    data_raw = []
    coe1 = []
    coe2 = []


    _line = _line.strip('\r\n')
    _line += ' '
    line_content = _line.split(' ')
    # print("line:", line_content)

    if _labels_type:
        for i in range(len(line_content) - 1):
            data_raw.append(line_content[i])

        for i in range(3):
            coe1.append(data_raw[i])

        for i in range(3, 6):
            coe2.append(data_raw[i])

    else:
        for i in range(7, 10):
            coe1.append(line_content[i])

        for i in range(10, 13):
            coe2.append(line_content[i])

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


def evaluateLane(_image_gt_mask, _image_to_display, x_list, y_list, scan_limit, text):
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
            print("mask:", _image_gt_mask[y_list[i], x_list[i]])
            if np.amax(_image_gt_mask[y_list[i], x_list[i]]) > 100:
                # print("--")
                cv2.circle(_image_to_display, (x_list[i], y_list[i]), 4, (255, 255, 0), thickness=-1, lineType=1, shift=0)
                counter_TP += 1

            else:
                # print("no")
                cv2.circle(_image_to_display, (x_list[i], y_list[i]), 4, (100, 100, 100), thickness=-1, lineType=1, shift=0)
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

if normalized_dataset:
    text_file = open(normalized_file_name, "r")
    normalization_params = text_file.read().split()
    text_file.close()

tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_jpeg(tf.read_file(file_input))
images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128. - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobileNet_v2.training_scope(is_training=False)):
    logits, endpoints = mobileNet_v2.mobilenet(images, output_regression_count)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, checkpoint)

    for i in range(len(file_list)):
        with open(src_folder + '/' + file_list[i] + '.txt') as f:
            line = f.readline()
            # print(line)

        coe_gt_1, coe_gt_r = get_two_lane_labels(line, default_label_type)
        # print("coe1:", coe_gt_1)
        # print("coe2:", coe_gt_r)

        image = cv2.imread(src_folder + '/' + file_list[i] + '.jpg')
        img_h = image.shape[0]
        img_w = image.shape[1]
        # blank_image = image
        height_hat = img_w

        # do prediction
        x = logits.eval(feed_dict={file_input: src_folder + '/' + file_list[i] + '.jpg'})
        x = x[0]
        # print(x)
        unnormalizeFromParams(x, normalization_params)
        predicted_left_lane = x[:3]
        predicted_right_lane = x[3:]
        # print("L:", predicted_left_lane, "R:", predicted_right_lane)

        # mask for ground-truth (gt) and predicted (pd)
        mask_image_gt_left_lane = np.zeros([img_h, img_w, 3], np.uint8)
        mask_image_gt_right_lane = np.zeros([img_h, img_w, 3], np.uint8)

        mask_image_pd_left_lane = np.zeros([img_h, img_w, 3], np.uint8)
        mask_image_pd_right_lane = np.zeros([img_h, img_w, 3], np.uint8)

        drawOneLane(mask_image_gt_left_lane, coe_gt_1, "red", height_hat, c_radius=10)
        drawOneLane(image, coe_gt_1, "red", height_hat, c_radius=10)
        drawOneLane(mask_image_gt_right_lane, coe_gt_r, "green", height_hat, c_radius=10)
        drawOneLane(image, coe_gt_r, "green", height_hat, c_radius=10)

        drawOneLane(mask_image_pd_left_lane, predicted_left_lane, "white", height_hat)
        # drawOneLane(mask_image_gt_left_lane, coeL_test, "white", height_hat)
        drawOneLane(mask_image_pd_right_lane, predicted_right_lane, "white", height_hat)
        # drawOneLane(mask_image_gt_right_lane, coeR_test, "white", height_hat)

        # channel doesn't matter since pd line is white
        print("get L:")
        xl, yl = get_mask_coordinates(mask_image_pd_left_lane, channel='B', val=255)
        print("get L:")
        xr, yr = get_mask_coordinates(mask_image_pd_right_lane, channel='B', val=255)

        Ltp, Lfp = evaluateLane(mask_image_gt_left_lane,image, xl, yl, scan_limit=img_h / 2, text="left")
        Rtp, Rfp = evaluateLane(mask_image_gt_right_lane,image, xr, yr, scan_limit=img_h / 2, text="right")

        LeftPrecision = Ltp / (Ltp + Lfp)
        RightPrecision = Rtp / (Rtp + Rfp)

        average_precision = (LeftPrecision + RightPrecision) / 2

        average_list.append(average_precision)

        # print(img_w / 2)
        # print("xl:", xl, "yl:", yl)
        # print("xl:", xr, "yl:", yr)

        # image = cv2.resize(image, None, fx=0.5, fy=0.5)
        # image = cv2.resize(image, (960, 540))
        # print(image.shape)
        # cv2.imshow("image", image)
        # cv2.imshow("L-mask", mask_image_gt_left_lane)
        # cv2.imshow("R-mask", mask_image_gt_right_lane)
        # cv2.imshow("pd-mask", mask_image_pd_left_lane)
        # keysave = cv2.waitKey(0) & 0xFF

total_average = sum(average_list) / len(average_list)
print("total precision of folder:", total_average)

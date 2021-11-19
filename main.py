from roifile import ImagejRoi
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham
import scipy
from scipy import signal
import pandas as pd

# Pixel resolution px/um
PIXEL_RESOLUTION = 16.129


def get_data(channel, img_path, roi_number):
    channel = channel
    image_path = img_path

    roi_list = sorted(os.listdir('RoiSet'))
    roi = ImagejRoi.fromfile(f'RoiSet/{roi_list[roi_number]}')
    print(roi)

    x_1 = int(roi.x1)
    x_2 = int(roi.x2)
    y_1 = int(roi.y1)
    y_2 = int(roi.y2)

    px_array = list(bresenham(x_1, y_1, x_2, y_2))

    im = Image.open(image_path)
    im = im.convert('RGB')
    pixel_map = im.load()

    y_array = []
    x_array = np.arange(0, len(px_array))
    # x_array = np.flipud(x_array)

    for i in range(im.width):
        for j in range(im.height):
            if (i, j) in px_array:
                if channel == 'red':
                    intensity = pixel_map[i, j][0]
                else:
                    intensity = pixel_map[i, j][1]
                y_array.append(intensity)

    smooth_y = scipy.signal.savgol_filter(y_array, 51, 10)

    return x_array, y_array, smooth_y


number_of_rois = len(os.listdir('RoiSet'))

for r in range(number_of_rois):
    x_green, y_green, s_green = get_data('green', 'caspr.png', r)
    x_red, y_red, s_red = get_data('red', 'nav.png', r)

    # Sometimes there are local minima at the edges... the below code re-slices s_green to focus on the middle
    caspr_trough_index = np.where(s_green == min(s_green[int(len(s_green) * .25): int(len(s_green) * .75)]))
    caspr_trough_distance = (x_green / PIXEL_RESOLUTION)[caspr_trough_index]

    left_perinode_max_index = np.where(s_green == max(s_green[caspr_trough_index[0][0]:]))[0][0]
    left_perinode_distance = (x_green / PIXEL_RESOLUTION)[left_perinode_max_index]

    right_perinode_max_index = np.where(s_green == max(s_green[0:caspr_trough_index[0][0]]))[0][0]
    right_perinode_distance = (x_green / PIXEL_RESOLUTION)[right_perinode_max_index]

    average_maxima = (s_green[left_perinode_max_index] + s_green[right_perinode_max_index]) / 2
    minima = s_green[caspr_trough_index][0]
    threshold = minima + ((average_maxima - minima) / 2)

    critical_points_x = [caspr_trough_index[0] / PIXEL_RESOLUTION,
                         left_perinode_max_index / PIXEL_RESOLUTION,
                         right_perinode_max_index / PIXEL_RESOLUTION]
    critical_points_y = [s_green[caspr_trough_index][0], s_green[left_perinode_max_index],
                         s_green[right_perinode_max_index]]

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.set_facecolor("#949494")
    plt.plot(x_green / PIXEL_RESOLUTION, s_green, label='Caspr1', color='#296600')
    plt.plot(x_red / PIXEL_RESOLUTION, s_red, label='Nav1.6', color='#dd0000')
    plt.scatter(critical_points_x, critical_points_y)
    plt.plot(x_green / PIXEL_RESOLUTION, np.full(x_green.shape, threshold), color='#ec9706', label='Threshold')

    plt.ylabel('8-bit Intensity', fontsize=14)
    plt.xlabel('Distance in Microns', fontsize=14)
    plt.title('Caspr1 & Nav1.6 Intensity Distributions', fontsize=16)

    plt.legend()
    plt.show()

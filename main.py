from roifile import ImagejRoi
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham
import scipy
from scipy import signal
from scipy import integrate
import pandas as pd

# Pixel resolution px/um
PIXEL_RESOLUTION = 16.129


def get_data(channel, img_path, roi_number):
    channel = channel
    image_path = img_path

    roi_list = sorted(os.listdir('RoiSet'))
    roi = ImagejRoi.fromfile(f'RoiSet/{roi_list[roi_number]}')

    x_1 = int(roi.x1)
    x_2 = int(roi.x2)
    y_1 = int(roi.y1)
    y_2 = int(roi.y2)

    # Linear interpolation of pixels isn't as straightforward as y=mx+b.
    # Bresenham's algorithm determines the pixels to color to render a line between two points.
    px_array = list(bresenham(x_1, y_1, x_2, y_2))

    im = Image.open(image_path)
    im = im.convert('RGB')
    pixel_map = im.load()

    intensity_array = []
    distance_array = np.arange(0, len(px_array))
    # distance_array = np.flipud(distance_array)

    for i in range(im.width):
        for j in range(im.height):
            if (i, j) in px_array:
                if channel == 'red':
                    intensity = pixel_map[i, j][0]
                else:
                    intensity = pixel_map[i, j][1]
                intensity_array.append(intensity)

    smooth_y = scipy.signal.savgol_filter(intensity_array, 51, 10)

    return distance_array, intensity_array, smooth_y, roi.name


number_of_rois = len(os.listdir('RoiSet'))

for r in range(number_of_rois):
    x_green, y_green, s_green, roi_name_1 = get_data('green', 'caspr.png', r)
    x_red, y_red, s_red, roi_name_2 = get_data('red', 'nav.png', r)

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

    nav_peak_index = np.where(s_red == max(s_red[int(len(s_red) * .25): int(len(s_red) * .75)]))
    nav_peak_distance = (x_red / PIXEL_RESOLUTION)[nav_peak_index]

    # Get Bounds of peri-nodes
    # [Right Side]: starts at trough and moves rightward. First cross below threshold stops it.

    hit_min = False
    cross_up = False
    bounds_array = []

    for entry in s_green[caspr_trough_index[0][0]:]:
        if entry == min(s_green[int(len(s_green) * .25): int(len(s_green) * .75)]):
            hit_min = True
        if entry >= threshold and hit_min:
            cross_up = True
            right_perinode_bound = np.where(s_green == entry)[0][0]
            bounds_array.append(right_perinode_bound)
        if entry <= threshold and cross_up:
            break

    right_perinode_bound1 = bounds_array[0]
    right_perinode_bound2 = bounds_array[-1]

    # [Left Side]: Same idea as for the right side. Flipped array so that it starts at the minima
    cross_up = False
    bounds_array2 = []
    s_green_from_min_to_start = np.flipud(s_green[0:caspr_trough_index[0][0]])
    for entry in s_green_from_min_to_start:
        if entry >= threshold:
            cross_up = True
            left_perinode_bound = np.where(s_green == entry)[0][0]
            bounds_array2.append(left_perinode_bound)
        if entry <= threshold and cross_up:
            break

    left_perinode_bound1 = bounds_array2[0]
    left_perinode_bound2 = bounds_array2[-1]

    # Critical Points for Scatter Plot Visualization
    critical_points_x = [caspr_trough_index[0] / PIXEL_RESOLUTION, left_perinode_max_index / PIXEL_RESOLUTION,
                         right_perinode_max_index / PIXEL_RESOLUTION, right_perinode_bound1 / PIXEL_RESOLUTION,
                         right_perinode_bound2 / PIXEL_RESOLUTION, left_perinode_bound1 / PIXEL_RESOLUTION,
                         left_perinode_bound2 / PIXEL_RESOLUTION]
    critical_points_y = [s_green[caspr_trough_index][0], s_green[left_perinode_max_index],
                         s_green[right_perinode_max_index], threshold, threshold, threshold, threshold]

    red_peak_x = [nav_peak_index[0] / PIXEL_RESOLUTION]
    red_peak_y = [s_red[nav_peak_index][0]]

    # Get Node Measurements
    left_perinode_length = abs(left_perinode_bound1 - left_perinode_bound2) / PIXEL_RESOLUTION
    right_perinode_length = abs(right_perinode_bound1 - right_perinode_bound2) / PIXEL_RESOLUTION

    average_perinode_length = round((left_perinode_length + right_perinode_length) / 2, 2)
    node_length = round(abs(right_perinode_bound1 - left_perinode_bound1) / PIXEL_RESOLUTION, 2)
    node_shift = round(abs(nav_peak_distance - caspr_trough_distance)[0], 2)

    results_list = [f'{roi_name_1}', average_perinode_length, node_length, node_shift]
    results_dict = {
        'ROI Name': f'{roi_name_1}',
        'Avg. Perinode Length (um)': average_perinode_length,
        'Node Length (um)': node_length,
        'Node Shift (um)': node_shift
    }

    try:
        df = df.append(results_dict, ignore_index=True)
    except NameError:
        df = pd.DataFrame(results_list).T
        df.columns = ['ROI Name', 'Avg. Perinode Length (um)', 'Node Length (um)', 'Node Shift (um)']

    # area_perinode_left = integrate.simpson(s_green[bounds] - threshold[bounds], x_array[bounds])

    # Plotting and Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.set_facecolor("#949494")
    plt.plot(x_green / PIXEL_RESOLUTION, s_green, label='Caspr1', color='#296600')
    plt.plot(x_red / PIXEL_RESOLUTION, s_red, label='Nav1.6', color='#dd0000')
    plt.scatter(critical_points_x, critical_points_y, color='green')
    plt.scatter(red_peak_x, red_peak_y, color='red')
    plt.plot(x_green / PIXEL_RESOLUTION, np.full(x_green.shape, threshold), color='#ec9706', label='Threshold')

    plt.ylabel('8-bit Intensity', fontsize=14)
    plt.xlabel('Distance in Microns', fontsize=14)
    plt.title('Caspr1 & Nav1.6 Intensity Distributions', fontsize=16)

    plt.legend()
    plt.savefig(f'figures/{roi_name_1}.png')

df.to_excel('Results.xlsx')

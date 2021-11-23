from roifile import ImagejRoi
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham
import scipy
from scipy import signal
import re
import pandas as pd
from pathlib import Path

# Pixel resolution px/um
PIXEL_RESOLUTION = 16.129


def get_data(channel, img_path, roi_number):
    channel = channel
    image_path = img_path

    roi_list = sorted(os.listdir(ROI_directory))
    roi = ImagejRoi.fromfile(ROI_directory + f'/{roi_list[roi_number]}')

    # Linear interpolation of pixels isn't as straightforward as y=mx+b.
    # Bresenham's algorithm determines the pixels to color to render a line between two points.
    # The exception handling below is due to some people using segmented lines and other the straight line.

    try:
        length = len(roi.subpixel_coordinates)
        if length == 2:
            # segment_type = 'segmented line'
            coordinates = roi.subpixel_coordinates
            x1 = round(coordinates[0][0], 0)
            x2 = round(coordinates[1][0], 0)
            y1 = round(coordinates[0][1], 0)
            y2 = round(coordinates[1][1], 0)
        elif length == 3:
            # segment_type = 'branched segment'
            coordinates = roi.subpixel_coordinates
            x1 = round(coordinates[0][0], 0)
            x2 = round(coordinates[1][0], 0)
            x3 = round(coordinates[2][0], 0)
            y1 = round(coordinates[0][1], 0)
            y2 = round(coordinates[1][1], 0)
            y3 = round(coordinates[2][1], 0)
        elif length == 4:
            # segment_type = 'extra branched segment'
            coordinates = roi.subpixel_coordinates
            x1 = round(coordinates[0][0], 0)
            x2 = round(coordinates[1][0], 0)
            x3 = round(coordinates[2][0], 0)
            x4 = round(coordinates[3][0], 0)
            y1 = round(coordinates[0][1], 0)
            y2 = round(coordinates[1][1], 0)
            y3 = round(coordinates[2][1], 0)
            y4 = round(coordinates[3][1], 0)
        else:
            # segment_type = 'other'
            pass
    except TypeError:
        # segment_type = 'true line'
        x1 = round(roi.x1, 0)
        x2 = round(roi.x2, 0)
        y1 = round(roi.y1, 0)
        y2 = round(roi.y2, 0)

    px_array_1 = list(bresenham(int(x1), int(y1), int(x2), int(y2)))
    px_array_2 = []
    px_array_3 = []

    try:
        px_array_2 = list(bresenham(int(x2), int(y2), int(x3), int(y3)))
    except NameError:
        pass
    try:
        px_array_3 = list(bresenham(int(x3), int(y3), int(x4), int(y4)))
    except NameError:
        pass

    px_array = px_array_1 + px_array_2 + px_array_3

    im = Image.open(image_path)
    im = im.convert('RGB')
    pixel_map = im.load()

    intensity_array = []

    for i in range(im.width):
        for j in range(im.height):
            if (i, j) in px_array:
                if channel == 'red':
                    intensity = pixel_map[i, j][0]
                else:
                    intensity = pixel_map[i, j][1]
                intensity_array.append(intensity)

    distance_array = np.arange(0, len(intensity_array))
    # distance_array = np.flipud(distance_array)

    try:
        smooth_y = scipy.signal.savgol_filter(intensity_array, 51, 10)
    except ValueError:
        length_of_intensity_array = len(intensity_array)
        if length_of_intensity_array % 2 == 0:
            smooth_y = scipy.signal.savgol_filter(intensity_array, length_of_intensity_array - 3, 10)
        else:
            smooth_y = scipy.signal.savgol_filter(intensity_array, length_of_intensity_array - 2, 10)

    return distance_array, intensity_array, smooth_y, roi.name


def process_all_rois_for_an_image(caspr_image_path, nav_image_path):
    number_of_rois = len(os.listdir(ROI_directory))

    for r in range(number_of_rois):
        keep_datapoint = True
        try:
            x_green, y_green, s_green, roi_name_1 = get_data('green', caspr_image_path, r)
            x_red, y_red, s_red, roi_name_2 = get_data('red', nav_image_path, r)

            # Sometimes there are local minima at the edges... the below code re-slices s_green to focus on the middle
            caspr_trough_index = np.where(s_green == min(s_green[int(len(s_green) * .25): int(len(s_green) * .75)]))
            caspr_trough_distance = (x_green / PIXEL_RESOLUTION)[caspr_trough_index]
            caspr_trough_intensity = s_green[caspr_trough_index][0]

            left_perinode_max_index = np.where(s_green == max(s_green[caspr_trough_index[0][0]:]))[0][0]
            left_perinode_distance = (x_green / PIXEL_RESOLUTION)[left_perinode_max_index]
            left_perinode_intensity = s_green[left_perinode_max_index]

            right_perinode_max_index = np.where(s_green == max(s_green[0:caspr_trough_index[0][0]]))[0][0]
            right_perinode_distance = (x_green / PIXEL_RESOLUTION)[right_perinode_max_index]
            right_perinode_intensity = s_green[right_perinode_max_index]

            average_maxima = (s_green[left_perinode_max_index] + s_green[right_perinode_max_index]) / 2
            minima = s_green[caspr_trough_index][0]
            lower_max = min(s_green[left_perinode_max_index], s_green[right_perinode_max_index])
            threshold = minima + ((lower_max - minima) / 2)

            average_perinode_intensity = round((left_perinode_intensity + right_perinode_intensity) / 2, 0)

            nav_peak_index = np.where(s_red == max(s_red[int(len(s_red) * .25): int(len(s_red) * .75)]))
            nav_peak_distance = (x_red / PIXEL_RESOLUTION)[nav_peak_index]
            nav_peak_intensity = round(s_red[nav_peak_index][0], 0)

            # Get Bounds of peri-nodes
            # [Right Side]: starts at trough and moves rightward. First cross below threshold stops it.

            # BUG FIX!: For some reason, it thinks the indices are backwards. So left_perinode_max index is really
            # right_perinode_max_index. Thats why the below code references the opposite side line 158, 174

            hit_min = False
            cross_up = False
            bounds_array = []

            for entry in s_green[caspr_trough_index[0][0]:]:
                if entry >= threshold:
                    cross_up = True
                    right_perinode_bound = np.where(s_green == entry)[0][0]
                    bounds_array.append(right_perinode_bound)
                if entry <= threshold and cross_up and np.where(s_green == entry)[0][0] >= left_perinode_max_index:
                    # print(entry, threshold, np.where(s_green == entry)[0][0], right_perinode_max_index)
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
                if entry <= threshold and cross_up and np.where(s_green == entry)[0][0] <= right_perinode_max_index:
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

            if node_shift > 0.65 * node_length:
                keep_datapoint = False

            try:
                if left_perinode_length > right_perinode_length:
                    perinode_asymmetry = round(left_perinode_length / right_perinode_length, 2)
                else:
                    perinode_asymmetry = round(right_perinode_length / left_perinode_length, 2)
            except RuntimeWarning:
                perinode_asymmetry = 'Zero Division Error'

            image_identity = re.split('C1_|.png', caspr_image_path)[1]

            results_list = [image_identity, f'{roi_name_1}', average_perinode_length, node_length, node_shift,
                            average_perinode_intensity, nav_peak_intensity, perinode_asymmetry]

            results_dict = {
                'Image Identity': image_identity,
                'ROI Name': f'{roi_name_1}',
                'Avg. Perinode Length (um)': average_perinode_length,
                'Node Length (um)': node_length,
                'Node Shift (um)': node_shift,
                'Avg. Peak Perinode Intensity': average_perinode_intensity,
                'Peak NaV Intensity': nav_peak_intensity,
                'Perinode Asymmetry': perinode_asymmetry
            }

            if keep_datapoint:
                try:
                    df = df.append(results_dict, ignore_index=True)
                except NameError:
                    df = pd.DataFrame(results_list).T
                    df.columns = ['Image Identity', 'ROI Name', 'Avg. Perinode Length (um)', 'Node Length (um)',
                                  'Node Shift (um)', 'Avg. Peak Perinode Intensity', 'Peak NaV Intensity',
                                  'Perinode Asymmetry']

                # area_perinode_left = integrate.simpson(s_green[bounds] - threshold[bounds], x_array[bounds])

                # Plotting and Visualization
                fig = plt.figure(figsize=(12, 8))
                ax = plt.axes()
                ax.set_facecolor("#7c8594")
                plt.plot(x_green / PIXEL_RESOLUTION, s_green, label='Caspr1', color='#0e1d35')
                # plt.scatter(x_green / PIXEL_RESOLUTION, s_green, color='orange')
                # plt.plot(x_green / PIXEL_RESOLUTION, y_green, color='red')
                plt.plot(x_red / PIXEL_RESOLUTION, s_red, label='Nav 1.6', color='#dddee5')
                plt.scatter(critical_points_x, critical_points_y, color='#1167b1', label='Caspr1 Critical Points')
                plt.scatter(red_peak_x, red_peak_y, color='#d2b48c', label='Nav 1.6 Critical Point')
                plt.plot(x_green / PIXEL_RESOLUTION, np.full(x_green.shape, threshold), color='green',
                         label='Threshold')

                plt.ylabel('8-bit Intensity', fontsize=14)
                plt.xlabel('Distance in Microns', fontsize=14)
                plt.title(f'Caspr1 & Nav1.6 Intensity Distributions\n ROI:{roi_name_1}', fontsize=16)

                plt.legend(facecolor='#7c8594')
                Path(f"figures/{caspr_image_path.split('/')[-1].strip('.png')}").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"figures/{caspr_image_path.split('/')[-1].strip('.png')}/{roi_name_1}.png")
                plt.close(fig)
        except Exception as e:
            try:
                print(f'Error with {roi_name_1}: {e}')
            except UnboundLocalError:
                print(f'Error with Unknown ROI: {e}')

    try:
        df.to_excel(f'Excel Output/individual images/{caspr_image_path.split("/")[-1].strip(".png").strip("C1")}_Results.xlsx')
    except UnboundLocalError:
        pass

    try:
        return df
    except UnboundLocalError:
        return pd.DataFrame(columns=['Image Identity', 'ROI Name', 'Avg. Perinode Length (um)',
                                     'Node Length (um)', 'Node Shift (um)',
                                     'Avg. Peak Perinode Intensity', 'Peak NaV Intensity',
                                     'Perinode Asymmetry'])


def process_animal(animal_identity, eye_side):
    file_and_folder_array = os.listdir(f'{animal_identity}/{eye_side}')
    list_of_caspr_png = []
    list_of_nav_png = []

    for element in file_and_folder_array:
        if '.png' and 'C1_' in element:
            list_of_caspr_png.append(element)
        if '.png' and 'C2_' in element:
            list_of_nav_png.append(element)
    zipped_caspr_nav_filenames = zip(list_of_caspr_png, list_of_nav_png)

    whole_animal_dataframe = pd.DataFrame(columns=['Image Identity', 'ROI Name', 'Avg. Perinode Length (um)',
                                                   'Node Length (um)', 'Node Shift (um)',
                                                   'Avg. Peak Perinode Intensity', 'Peak NaV Intensity',
                                                   'Perinode Asymmetry'])

    for (caspr, nav) in zipped_caspr_nav_filenames:
        ROI_folder_name = f'RoiSet#{caspr.strip(".png")}'
        global ROI_directory
        ROI_directory = f'{animal_identity}/{eye_side}/{ROI_folder_name}'
        single_image_dataframe = process_all_rois_for_an_image(caspr_image_path=f'{animal_identity}/{eye_side}/{caspr}',
                                                               nav_image_path=f'{animal_identity}/{eye_side}/{nav}')
        whole_animal_dataframe = pd.concat([whole_animal_dataframe, single_image_dataframe], ignore_index=True)

    Path(f"Excel Output/{animal_identity}").mkdir(parents=True, exist_ok=True)
    whole_animal_dataframe.to_excel(f'Excel Output/{animal_identity}/{eye_side}_overall_results.xlsx')


# process_animal('12', 'Left')
# process_animal('12', 'Right')

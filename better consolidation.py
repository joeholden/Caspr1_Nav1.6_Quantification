import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Gets all the file paths for excel sheets
all_file_paths = []
mid_roi_paths = []
eye_roi_paths = []
chiasm_roi_paths = []

for directory_path, subdirectory, files in os.walk('C:/Users/JoeHo/Downloads/caspr results/caspr nav results/Excel '
                                                   'Output/individual images'):
    for file in files:
        if file.endswith('.xlsx'):
            all_file_paths.append(os.path.join(directory_path, file))

columns = ['Image Identity', 'ROI Name', 'Avg. Perinode Length (um)', 'Node Length (um)',
           'Node Shift (um)', 'Avg. Peak Perinode Intensity', 'Peak NaV Intensity',
           'Perinode Asymmetry']

# Creates Empty DataFrame
composite_df_left_c_cs = pd.DataFrame(columns=columns)
composite_df_left_c_cn = composite_df_left_c_cs.copy()
composite_df_left_c_mom = composite_df_left_c_cs.copy()
composite_df_left_c_wt = composite_df_left_c_cs.copy()

composite_df_right_c_cs = composite_df_left_c_cs.copy()
composite_df_right_c_cn = composite_df_left_c_cs.copy()
composite_df_right_c_mom = composite_df_left_c_cs.copy()
composite_df_right_c_wt = composite_df_left_c_cs.copy()

composite_df_left_e_cs = composite_df_left_c_cs.copy()
composite_df_left_e_cn = composite_df_left_c_cs.copy()
composite_df_left_e_mom = composite_df_left_c_cs.copy()
composite_df_left_e_wt = composite_df_left_c_cs.copy()

composite_df_right_e_cs = composite_df_left_c_cs.copy()
composite_df_right_e_cn = composite_df_left_c_cs.copy()
composite_df_right_e_mom = composite_df_left_c_cs.copy()
composite_df_right_e_wt = composite_df_left_c_cs.copy()

composite_df_left_m_cs = composite_df_left_c_cs.copy()
composite_df_left_m_cn = composite_df_left_c_cs.copy()
composite_df_left_m_mom = composite_df_left_c_cs.copy()
composite_df_left_m_wt = composite_df_left_c_cs.copy()

composite_df_right_m_cs = composite_df_left_c_cs.copy()
composite_df_right_m_cn = composite_df_left_c_cs.copy()
composite_df_right_m_mom = composite_df_left_c_cs.copy()
composite_df_right_m_wt = composite_df_left_c_cs.copy()

locations_dict = {
    '(C)': 'chiasm',
    '(M1)': 'mid',
    '(M2)': 'mid',
    '(E)': 'eye',
}

group_dictionary = {
    '_1 ': 'cs',
    '_7 ': 'cs',
    '_8 ': 'cs',
    '_9 ': 'cs',
    '_12': 'cn',
    '_13': 'cn',
    '_14': 'cn',
    '_15': 'cs',
    '_351': 'mom',
    '_352': 'mom',
    '_353': 'mom',
    '_354': 'mom',
    '_355': 'mom',
    '_WT1': 'wt',
    '_WT2': 'wt',
    '_wt3': 'wt',

}

locations_list = ['(C)', '(M1)', '(M2)', '(M)', '(E)']

for group in locations_list:
    for f in all_file_paths:
        for key in group_dictionary.keys():
            if key.lower() in f.lower():
                animal_key = key
                break
        if (group in f.upper()) and ('left' in f.lower()):
            print(f)
            if group_dictionary[animal_key] == 'cs':
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_left_m_cs = pd.concat([composite_df_left_m_cs, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_left_c_cs = pd.concat([composite_df_left_c_cs, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_left_e_cs = pd.concat([composite_df_left_e_cs, df], ignore_index=True)
            if group_dictionary[animal_key] == 'cn':
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_left_m_cn = pd.concat([composite_df_left_m_cn, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_left_c_cn = pd.concat([composite_df_left_c_cn, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_left_e_cn = pd.concat([composite_df_left_e_cn, df], ignore_index=True)
            if group_dictionary[animal_key] == 'mom':
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_left_m_mom = pd.concat([composite_df_left_m_mom, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_left_c_mom = pd.concat([composite_df_left_c_mom, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_left_e_mom = pd.concat([composite_df_left_e_mom, df], ignore_index=True)
            if group_dictionary[animal_key] == 'wt':
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_left_m_wt = pd.concat([composite_df_left_m_wt, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_left_c_wt = pd.concat([composite_df_left_c_wt, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_left_e_wt = pd.concat([composite_df_left_e_wt, df], ignore_index=True)

        if (group in f.upper()) and ('right' in f.lower()):
            print(f)
            if group_dictionary[animal_key] == 'cs':
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_right_m_cs = pd.concat([composite_df_right_m_cs, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_right_c_cs = pd.concat([composite_df_right_c_cs, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_right_e_cs = pd.concat([composite_df_right_e_cs, df], ignore_index=True)
            if group_dictionary[animal_key] == 'cn':
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_right_m_cn = pd.concat([composite_df_right_m_cn, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_right_c_cn = pd.concat([composite_df_right_c_cn, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_right_e_cn = pd.concat([composite_df_right_e_cn, df], ignore_index=True)
            if group_dictionary[animal_key] == 'mom':
                print('mom')
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_right_m_mom = pd.concat([composite_df_right_m_mom, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_right_c_mom = pd.concat([composite_df_right_c_mom, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_right_e_mom = pd.concat([composite_df_right_e_mom, df], ignore_index=True)
            if group_dictionary[animal_key] == 'wt':
                if group == '(M1)' or group == '(M2)' or group == '(M)':
                    df = pd.read_excel(f)
                    composite_df_right_m_wt = pd.concat([composite_df_right_m_wt, df], ignore_index=True)
                elif group == '(C)':
                    df = pd.read_excel(f)
                    composite_df_right_c_wt = pd.concat([composite_df_right_c_wt, df], ignore_index=True)
                else:
                    df = pd.read_excel(f)
                    composite_df_right_e_wt = pd.concat([composite_df_right_e_wt, df], ignore_index=True)


def make_excel():
    composite_df_left_e_mom.to_excel('(E)_MOM_Left.xlsx')
    composite_df_left_e_wt.to_excel('(E)_WT_Left.xlsx')
    composite_df_left_e_cn.to_excel('(E)_CN_Left.xlsx')
    composite_df_left_e_cs.to_excel('(E)_CS_Left.xlsx')

    composite_df_right_e_mom.to_excel('(E)_MOM_Right.xlsx')
    composite_df_right_e_wt.to_excel('(E)_WT_Right.xlsx')
    composite_df_right_e_cn.to_excel('(E)_CN_Right.xlsx')
    composite_df_right_e_cs.to_excel('(E)_CS_Right.xlsx')

    composite_df_left_c_mom.to_excel('(C)_MOM_Left.xlsx')
    composite_df_left_c_wt.to_excel('(C)_WT_Left.xlsx')
    composite_df_left_c_cn.to_excel('(C)_CN_Left.xlsx')
    composite_df_left_c_cs.to_excel('(C)_CS_Left.xlsx')

    composite_df_right_c_mom.to_excel('(C)_MOM_Right.xlsx')
    composite_df_right_c_wt.to_excel('(C)_WT_Right.xlsx')
    composite_df_right_c_cn.to_excel('(C)_CN_Right.xlsx')
    composite_df_right_c_cs.to_excel('(C)_CS_Right.xlsx')

    composite_df_left_m_mom.to_excel('(M)_MOM_Left.xlsx')
    composite_df_left_m_wt.to_excel('(M)_WT_Left.xlsx')
    composite_df_left_m_cn.to_excel('(M)_CN_Left.xlsx')
    composite_df_left_m_cs.to_excel('(M)_CS_Left.xlsx')

    composite_df_right_m_mom.to_excel('(M)_MOM_Right.xlsx')
    composite_df_right_m_wt.to_excel('(M)_WT_Right.xlsx')
    composite_df_right_m_cn.to_excel('(M)_CN_Right.xlsx')
    composite_df_right_m_cs.to_excel('(M)_CS_Right.xlsx')


def plot():

    regions = ['C', 'M', 'E']
    columns_abbreviated = ['Avg. Perinode Length (um)', 'Node Length (um)', 'Node Shift (um)',
                           'Avg. Peak Perinode Intensity', 'Peak NaV Intensity', 'Perinode Asymmetry']

    for col in columns_abbreviated:
        # Chiasm

        bar_groups = ['Left CS', 'Right CS', 'Left CN', 'Right CN', 'Left MOM', 'Right MOM', 'Left N', 'Right N']
        x_position = np.arange(len(bar_groups))
        bar_means = [0] * 8
        bar_sem = [0] * 8

        # for column_heading in columns:
        bar_means[0] = np.mean(composite_df_left_c_cs[col])
        bar_means[1] = np.mean(composite_df_right_c_cs[col])
        bar_means[2] = np.mean(composite_df_left_c_cn[col])
        bar_means[3] = np.mean(composite_df_right_c_cn[col])
        bar_means[4] = np.mean(composite_df_left_c_mom[col])
        bar_means[5] = np.mean(composite_df_right_c_mom[col])
        bar_means[6] = np.mean(composite_df_left_c_wt[col])
        bar_means[7] = np.mean(composite_df_right_c_wt[col])

        bar_sem[0] = np.std(composite_df_left_c_cs[col]) / np.sqrt(np.size(composite_df_left_c_cs[col]))
        bar_sem[1] = np.std(composite_df_right_c_cs[col]) / np.sqrt(np.size(composite_df_right_c_cs[col]))
        bar_sem[2] = np.std(composite_df_left_c_cn[col]) / np.sqrt(np.size(composite_df_left_c_cn[col]))
        bar_sem[3] = np.std(composite_df_right_c_cn[col]) / np.sqrt(np.size(composite_df_right_c_cn[col]))
        bar_sem[4] = np.std(composite_df_left_c_mom[col]) / np.sqrt(np.size(composite_df_left_c_mom[col]))
        bar_sem[5] = np.std(composite_df_right_c_mom[col]) / np.sqrt(np.size(composite_df_right_c_mom[col]))
        bar_sem[6] = np.std(composite_df_left_c_wt[col]) / np.sqrt(np.size(composite_df_left_c_wt[col]))
        bar_sem[7] = np.std(composite_df_right_c_wt[col]) / np.sqrt(np.size(composite_df_right_c_wt[col]))

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.bar(x_position, bar_means, yerr=bar_sem, align='center',
               color=['#663399', '#663399', '#8B74BD', '#8B74BD', '#4066E0', '#4066E0', '#22277A', '#22277A'], capsize=7)
        ax.set_xticks(x_position)
        ax.set_xticklabels(bar_groups, size=15)
        ax.set_ylabel(col, size=18)
        plt.title(f'{col} (Chiasm Side)', size=24)

        plt.tight_layout()
        plt.savefig(f'{col} (C).png')

        # Eye

        bar_groups = ['Left CS', 'Right CS', 'Left CN', 'Right CN', 'Left MOM', 'Right MOM', 'Left N', 'Right N']
        x_position = np.arange(len(bar_groups))
        bar_means = [0] * 8
        bar_sem = [0] * 8

        # for column_heading in columns:
        bar_means[0] = np.mean(composite_df_left_e_cs[col])
        bar_means[1] = np.mean(composite_df_right_e_cs[col])
        bar_means[2] = np.mean(composite_df_left_e_cn[col])
        bar_means[3] = np.mean(composite_df_right_e_cn[col])
        bar_means[4] = np.mean(composite_df_left_e_mom[col])
        bar_means[5] = np.mean(composite_df_right_e_mom[col])
        bar_means[6] = np.mean(composite_df_left_e_wt[col])
        bar_means[7] = np.mean(composite_df_right_e_wt[col])

        bar_sem[0] = np.std(composite_df_left_e_cs[col]) / np.sqrt(np.size(composite_df_left_e_cs[col]))
        bar_sem[1] = np.std(composite_df_right_e_cs[col]) / np.sqrt(np.size(composite_df_right_e_cs[col]))
        bar_sem[2] = np.std(composite_df_left_e_cn[col]) / np.sqrt(np.size(composite_df_left_e_cn[col]))
        bar_sem[3] = np.std(composite_df_right_e_cn[col]) / np.sqrt(np.size(composite_df_right_e_cn[col]))
        bar_sem[4] = np.std(composite_df_left_e_mom[col]) / np.sqrt(np.size(composite_df_left_e_mom[col]))
        bar_sem[5] = np.std(composite_df_right_e_mom[col]) / np.sqrt(np.size(composite_df_right_e_mom[col]))
        bar_sem[6] = np.std(composite_df_left_e_wt[col]) / np.sqrt(np.size(composite_df_left_e_wt[col]))
        bar_sem[7] = np.std(composite_df_right_e_wt[col]) / np.sqrt(np.size(composite_df_right_e_wt[col]))

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.bar(x_position, bar_means, yerr=bar_sem, align='center',
               color=['#663399', '#663399', '#8B74BD', '#8B74BD', '#4066E0', '#4066E0', '#22277A', '#22277A'], capsize=7)
        ax.set_xticks(x_position)
        ax.set_xticklabels(bar_groups, size=15)
        ax.set_ylabel(col, size=18)
        plt.title(f'{col} (Eye Side)', size=24)

        plt.tight_layout()
        plt.savefig(f'{col} (E).png')

        # Mid

        bar_groups = ['Left CS', 'Right CS', 'Left CN', 'Right CN', 'Left MOM', 'Right MOM', 'Left N', 'Right N']
        x_position = np.arange(len(bar_groups))
        bar_means = [0] * 8
        bar_sem = [0] * 8

        # for column_heading in columns:
        bar_means[0] = np.mean(composite_df_left_m_cs[col])
        bar_means[1] = np.mean(composite_df_right_m_cs[col])
        bar_means[2] = np.mean(composite_df_left_m_cn[col])
        bar_means[3] = np.mean(composite_df_right_m_cn[col])
        bar_means[4] = np.mean(composite_df_left_m_mom[col])
        bar_means[5] = np.mean(composite_df_right_m_mom[col])
        bar_means[6] = np.mean(composite_df_left_m_wt[col])
        bar_means[7] = np.mean(composite_df_right_m_wt[col])

        bar_sem[0] = np.std(composite_df_left_m_cs[col]) / np.sqrt(np.size(composite_df_left_m_cs[col]))
        bar_sem[1] = np.std(composite_df_right_m_cs[col]) / np.sqrt(np.size(composite_df_right_m_cs[col]))
        bar_sem[2] = np.std(composite_df_left_m_cn[col]) / np.sqrt(np.size(composite_df_left_m_cn[col]))
        bar_sem[3] = np.std(composite_df_right_m_cn[col]) / np.sqrt(np.size(composite_df_right_m_cn[col]))
        bar_sem[4] = np.std(composite_df_left_m_mom[col]) / np.sqrt(np.size(composite_df_left_m_mom[col]))
        bar_sem[5] = np.std(composite_df_right_m_mom[col]) / np.sqrt(np.size(composite_df_right_m_mom[col]))
        bar_sem[6] = np.std(composite_df_left_m_wt[col]) / np.sqrt(np.size(composite_df_left_m_wt[col]))
        bar_sem[7] = np.std(composite_df_right_m_wt[col]) / np.sqrt(np.size(composite_df_right_m_wt[col]))

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.bar(x_position, bar_means, yerr=bar_sem, align='center',
               color=['#663399', '#663399', '#8B74BD', '#8B74BD', '#4066E0', '#4066E0', '#22277A', '#22277A'], capsize=7)
        ax.set_xticks(x_position)
        ax.set_xticklabels(bar_groups, size=15)
        ax.set_ylabel(col, size=18)
        plt.title(f'{col} (Mid)', size=24)

        plt.tight_layout()
        plt.savefig(f'{col} (M).png')


plot()

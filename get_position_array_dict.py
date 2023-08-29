import numpy as np
import pandas as pd
import os
import json


def get_array(file_path):
    # 从文件里获取数据并转化为ndarray
    f = pd.read_csv(file_path)
    x_values = f['x'].values
    y_values = f['y'].values
    density_values = f['normalized density'].values

    x_unique = np.sort(np.unique(x_values))
    y_unique = np.sort(np.unique(y_values))

    density_array = density_values.reshape(len(x_unique), len(y_unique))

    return density_array


def read_csv_files_in_folder(folder_path, multipler):
    # 处理整个文件夹内的文件
    arrays_dict = {}
    folder_list = os.listdir(folder_path)

    for folder_name in folder_list:
        subfolder_path = os.path.join(folder_path, folder_name, 'point_density')

        # Check if it is a folder
        if os.path.isdir(subfolder_path):
            array_list = []
            file_names = os.listdir(subfolder_path)
            file_names = sorted(file_names)

            for file_name in file_names:
                file_path = os.path.join(subfolder_path, file_name)
                if file_name.endswith('.csv'):
                    array = get_array(file_path)
                    array_list.append(array)

            mean_array = np.mean(array_list, axis=0)
            variance_array = np.var(array_list, axis=0)
            standard_array = np.std(array_list, axis=0)

            # Modify the result_array as per your requirements (e.g., no comparison)
            result_array = mean_array - multipler * standard_array

            arrays_dict[folder_name[0:12]] = result_array.tolist()

    return arrays_dict


if __name__ == '__main__':
    # parent_folder_path = 'D:/points/danglic/server_version/record_base'
    parent_folder_path = 'D:/points/test_with_speed'
    output_file = 'D:/points/danglic/server_version/arrays_data_test_1.4_multipler.json'
    multipler = 1.4  #衰减系数，越大误报率越低，但找到的实际区域就越小

    arrays_dict = read_csv_files_in_folder(parent_folder_path, multipler)

    # Write the dictionary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(arrays_dict, json_file)

    print("Data processing and writing to JSON completed.")

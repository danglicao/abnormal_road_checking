import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
def get_array(file_path):
    f = pd.read_csv(file_path)
    x_values = f['x'].values
    y_values = f['y'].values
    density_values = f['normalized density'].values

    x_unique = np.sort(np.unique(x_values))
    y_unique = np.sort(np.unique(y_values))

    density_array = density_values.reshape(len(x_unique), len(y_unique))

    return density_array

def read_csv_files_in_folder(folder_path, start_idx, end_idx):
    arrays_list = []
    file_names = os.listdir(folder_path)
    file_names = sorted(file_names)
    selected_files = file_names[start_idx:end_idx]

    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv'):
            array = get_array(file_path)
            arrays_list.append(array)

    return arrays_list

def read_compare_array(compare_folder, file_name):
    file_path = os.path.join(compare_folder, file_name)
    if file_path.endswith('.csv'):
        compare_array = get_array(file_path)
        return compare_array
    return None

def hot_map(array,index):
    data = array.astype(np.uint8)
    image = Image.fromarray(data)

    # plt.clf()
    # plt.title('Normalized Density')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # colors = [(1, 1, 1), (0, 0, 1)]  # 白色到蓝色的颜色渐变
    # cmap = LinearSegmentedColormap.from_list('white_blue', colors)
    # sns.heatmap(data=array, cmap='Blues', vmin=0, vmax=255)
    # # sns.heatmap(data=array, cmap=cmap)
    # plt.savefig(f'D:/points/save_fig/{filename}.png')
    new_size = (1920, 1080)
    image = image.resize(new_size)
    # image.save(f'D:/points/fig_test2/{index}.png')
    return image

def detect_and_draw_contours(image_array):
    # Convert the image array to a suitable data type (8-bit unsigned integer)
    image_array_uint8 = (image_array * 255).astype(np.uint8)

    # Convert the 8-bit unsigned integer image to a grayscale image
    gray_image = cv2.cvtColor(image_array_uint8, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to create a black and white image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding rectangles around each contour
    result_image = image_array.copy()  # Create a copy of the BGR image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result_image



def detect_and_draw_rectangles(image_array):
    # Convert the image to a NumPy array
    image_array_np = np.array(image_array)

    # Convert the image array to a suitable data type (8-bit unsigned integer)
    image_array_uint8 = (image_array_np * 255).astype(np.uint8)

    # Apply binary thresholding to create a black and white image
    _, binary_image = cv2.threshold(image_array_uint8, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around each contour (white point cluster)
    result_image = image_array_np.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the result back to PIL image format
    result_image_pil = Image.fromarray(result_image)

    return result_image_pil

def process_files_in_subfolders(parent_folder_path, output_folder_base, batch_size, multipler):
    folder_list = os.listdir(parent_folder_path)

    for folder_name in folder_list:
        folder_path = os.path.join(parent_folder_path, folder_name, 'point_density')
        compare_folder = folder_path

        # Check if it is a folder
        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_folder_base, folder_name, 'fig')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            total_files = len(os.listdir(folder_path))

            for start_idx in range(0, total_files - batch_size + 1):
                end_idx = start_idx + batch_size
                arrays_list = read_csv_files_in_folder(folder_path, start_idx, end_idx)

                mean_array = np.mean(arrays_list, axis=0)
                variance_array = np.var(arrays_list, axis=0)
                standard_array = np.std(arrays_list, axis=0)

                try:
                    compare_file_name = os.listdir(compare_folder)[end_idx + 1]
                    compare_array = read_compare_array(compare_folder, compare_file_name)

                    if compare_array is not None:
                        result_array = np.where(compare_array < mean_array - multipler * standard_array, 255, compare_array)

                        # Generate the hot map (PIL image)
                        hot_map_img = hot_map(result_array, start_idx)

                        # Draw rectangles around the white point clusters
                        result_image_with_rectangles = detect_and_draw_rectangles(hot_map_img)

                        # Save the image with rectangles
                        result_image_with_rectangles.save(os.path.join(output_folder, f'{end_idx}.png'))

                except IndexError:
                    print(f"IndexError occurred while processing {folder_name}. Skipping this folder...")
                    break  # Move to the next folder


if __name__ == '__main__':
    parent_folder_path = 'D:/points/test_with_speed'
    output_folder_base = 'D:/points/test_with_speed'
    batch_size = 30
    multipler = 2

    # for start_idx in range(0, total_files - batch_size + 1):
    #     end_idx = start_idx + batch_size
    #     arrays_list = read_csv_files_in_folder(folder_path, start_idx, end_idx)
    #
    #     mean_array = np.mean(arrays_list, axis=0)
    #     variance_array = np.var(arrays_list, axis=0)
    #     standard_array = np.std(arrays_list, axis=0)
    #
    #     compare_file_name = os.listdir(compare_folder)[start_idx]
    #     compare_array = read_compare_array(compare_folder, compare_file_name)
    #
    #     if compare_array is not None:
    #         result_array = np.where(compare_array < mean_array - standard_array, 255, compare_array)
    #
    #         # result_image = hot_map(result_array, start_idx)
    #         # Convert result_array to a BGR image format
    #         data = result_array.astype(np.uint8)
    #         result_image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    #
    #         # Detect and draw contours on the result image
    #         result_image_with_contours = detect_and_draw_contours(result_image)
    #
    #         # Save the image with contours
    #         cv2.imwrite(f'D:/points/fig_with_bound/{start_idx}.png', result_image_with_contours)

    process_files_in_subfolders(parent_folder_path, output_folder_base, batch_size, multipler)


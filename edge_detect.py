import numpy as np
import pandas as pd
import os
import json
from PIL import Image
import cv2
from shapely.geometry import Point, Polygon

def find_polygon_by_region_no(json_file, target_region_no):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'data' in data and isinstance(data['data'], list):
        for item in data['data']:
            if isinstance(item, dict) and 'region_no' in item and 'polygon' in item:
                if item['region_no'] == target_region_no:
                    return item['polygon']

    return None

def dilation_erosion(image):
    kernel_size = 3  # 设置结构元素大小
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 膨胀操作
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    # 腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return dilated_image

def connect_component(image):
    # 阈值化图像
    threshold_value = 100
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # 标记连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    # 分析连通区域并连接线条
    min_area_threshold = 20  # 设置一个最小的连通区域面积阈值
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_threshold:
            labels[labels == label] = 0  # 标记小的连通区域为背景

    # 创建新图像，连接线条后的结果
    connected_lines_image = np.zeros_like(image)
    connected_lines_image[labels > 0] = 255

    return connected_lines_image
def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def read_compare_array(compare_folder, file_name):
    file_path = os.path.join(compare_folder, file_name)
    if file_path.endswith('.csv'):
        compare_array = get_array(file_path)
        return compare_array
    return None

def get_array(file_path):
    f = pd.read_csv(file_path)
    x_values = f['x'].values
    y_values = f['y'].values
    density_values = f['normalized density'].values

    x_unique = np.sort(np.unique(x_values))
    y_unique = np.sort(np.unique(y_values))

    density_array = density_values.reshape(len(x_unique), len(y_unique))

    return density_array

def fig_format(array):
    data = array.astype(np.uint8)
    image = Image.fromarray(data)

    # new_size = (1920, 1080)
    # image = image.resize(new_size)
    # image.save(f'D:/points/fig_test2/{index}.png')
    return image

def get_absolute_coord(json_file, file_path):
    # region_no = os.path.basename(file_path)
    region_no = 'G32050700004'
    data_list = find_polygon_by_region_no(json_file, region_no)

    polygon_coords = []
    for point in data_list:
        polygon_coords.append((point["x"], point["y"]))
    polygon = Polygon(polygon_coords)
    min_x, min_y = polygon.bounds[0], polygon.bounds[1]
    max_x, max_y = polygon.bounds[2], polygon.bounds[3]
    # absolute_coords = [(x + min_x, y + min_y) for x, y in coords]
    return min_x, min_y, max_x, max_y

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


def process_compare_files(compare_folder, json_data, output_folder, region_json):
    # for folder_name, form_array in json_data.items():
        # output_folder = os.path.join(compare_folder, folder_name, 'fig')
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)

        for file_name in os.listdir(compare_folder):
            file_path = os.path.join(compare_folder, file_name)
            if file_name.endswith('.csv'):
                compare_array = get_array(file_path)  # Implement this function based on your needs
                compare_folder_name = os.path.basename(compare_folder) # Extract the last part of the compare folder path
                print(compare_folder_name)
                form_array = json_data.get(compare_folder_name[0:15])  # Get the form_array from the json_data
                print(np.max(form_array))
                result_array = np.where(compare_array < form_array, 255, compare_array)
                result_array = np.rot90(result_array, k=1)
                result_array = result_array.astype(np.uint8)
                # # 高斯滤波
                # blurred = cv2.GaussianBlur(result_array, (3, 3), 0)
                #
                # # 计算梯度
                # edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
                # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # wkt_dict = {}
                # min_x, min_y = get_absolute_coord(region_json, file_path)
                #
                # cv2.drawContours(blurred, contours, -1, (255, 255, 255), 2)
                #
                # # 显示绘制了轮廓的图像
                # cv2.imshow('Contours Image', blurred)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # for contour in contours:
                #     # contour = np.rot90(contour, k=3)
                #     for point in contour:
                #         x, y = point[0]  # 提取每个点的坐标 (x, y)
                #         print(f"Point: ({x+min_x}, {y+min_y})")
                #     print('a')

                # # 遍历每个轮廓
                # for idx, contour in enumerate(contours, start=1):
                #     # 将每个轮廓的 numpy 数组转换为 WKT 格式
                #     if len(contour) >= 3:
                #         wkt = "POLYGON ((" + ", ".join([f"{int(point[0])} {int(point[1])}" for point in contour.squeeze()]) + "))"
                #
                #         # 将 WKT 添加到字典中，以索引作为键
                #         wkt_dict[idx] = wkt

                # 将字典输出为 JSON 格式
                # print(edges)
                # # 显示结果
                # cv2.imshow('Original Image', result_array)
                # cv2.imshow('Canny Edges', edges)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # hot_map_img = fig_format(result_array)  # Implement this function based on your needs
                # print(np.max(hot_map_img))
                # result_image_with_rectangles = detect_and_draw_rectangles(hot_map_img)  # Implement this function based on your needs


                # if not os.path.exists(output_folder):
                #     os.makedirs(output_folder)
                #
                # file_path = os.path.join(output_folder, file_name + ".json")
                # with open(file_path, 'w') as json_file:
                #     json.dump(wkt_dict, json_file, indent=4)
                #
                # print("WKT contours saved in contours_wkt.json")


                # result_image_with_rectangles.save(os.path.join(output_folder, file_name + '.png'))
                #
                # # 应用中值滤波
                # filtered_array = connect_component(result_array.astype(np.uint8))
                # filtered_image = fig_format(filtered_array)
                # filtered_image.save(os.path.join(output_folder, file_name + 'filltered.png'))
                #
                # # 应用膨胀腐蚀
                dilated_array = dilation_erosion(result_array.astype(np.uint8))
                # dilated_image = fig_format(dilated_array)
                # dilated_image.save(os.path.join(output_folder, file_name + 'dilated.png'))
                #
                dilated_filterd_array = connect_component(dilated_array)
                # np.save('D:/points/danglic/array.npy', dilated_filterd_array)
                dilated_filterd_image = fig_format(dilated_filterd_array)
                dilated_filterd_image.save(os.path.join(output_folder, file_name + 'dialated_filltered_1.5.png'))
                # contours, hierarchy = cv2.findContours(dilated_filterd_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                wkt_dict = {}
                min_x, min_y, max_x, max_y = get_absolute_coord(region_json, file_path)
                edges = cv2.Canny(dilated_filterd_array, threshold1=100, threshold2=200)
                contours, hierarchy = cv2.findContours(dilated_filterd_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(dilated_filterd_array, contours, -1, (0, 255, 0), 2)
                #
                # 显示绘制了轮廓的图像
                cv2.imshow('Contours Image', dilated_filterd_array)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                contours_list = []
                for contour in contours:
                    contour_list = []
                    for point in contour:
                        # print(type(point))
                        x, y = point[0]  # 提取每个点的坐标 (x, y)
                        print(f"Point: ({x + min_x}, {max_y - y})")
                        contour_list.append((x + min_x, max_y - y))
                    print('a')
                    contours_list.append(contour_list)

                for contour in contours:
                    # contour = np.rot90(contour, k=3)
                    for point in contour:
                        x, y = point[0]  # 提取每个点的坐标 (x, y)
                        print(f"Point: ({x + min_x}, {y + min_y})")
                    print('a')
                # cv2.imshow('Canny Edges', dilated_filterd_array)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                wkt_dict = {}
                # 遍历每个轮廓
                # for idx, contour in enumerate(contours, start=1):
                #     # 将每个轮廓的 numpy 数组转换为 WKT 格式
                #     if len(contour) >= 3:
                #         # 获得每个轮廓的第一个点
                #         first_point = contour[0][0]
                #
                #         # 将每个轮廓的 numpy 数组转换为 WKT 格式
                #         wkt = "POLYGON ((" + ", ".join([f"{int(point[0])} {int(point[1])}" for point in
                #                                         contour.squeeze()]) + f", {int(first_point[0])} {int(first_point[1])}))"
                #
                #         # 将 WKT 添加到字典中，以索引作为键
                #         wkt_dict[idx] = wkt
                # print(contour_list)
                for idx, contour_position in enumerate(contours_list, start=1):
                # 将每个轮廓的 numpy 数组转换为 WKT 格式
                    if len(contour_position) >= 3:
                        # 获得每个轮廓的第一个点
                        first_point = contour_position[0]
                        wkt_coordinates = "POLYGON ((" + ", ".join([f"{x} {y}" for x, y in contour_position]) + f", {first_point[0]} {first_point[1]}))"
                        wkt_dict[idx] = wkt_coordinates
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                file_path = os.path.join(output_folder, file_name + ".json")
                with open(file_path, 'w') as json_file:
                    json.dump(wkt_dict, json_file, indent=4)

                print("WKT contours saved in contours_wkt.json")




if __name__ == '__main__':
    json_file_path = 'D:/points/danglic/server_version/arrays_data_test_1.5_multipler.json'  # Update this path
    compare_folder = 'D:/points/compare_data/G32050700004gt1'  # Update this path
    output_folder = 'D:/points/compare_data/G32050700004gt1/test_edge'
    region_json_file = 'D:/points/whole_data/分组信息.json'

    json_data = read_json_file(json_file_path)
    process_compare_files(compare_folder, json_data, output_folder, region_json_file)

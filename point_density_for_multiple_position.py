import os
from shapely.wkt import loads
from shapely.geometry import Point, Polygon
import json


def find_polygon_by_region_no(json_file, target_region_no):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'data' in data and isinstance(data['data'], list):
        for item in data['data']:
            if isinstance(item, dict) and 'region_no' in item and 'polygon' in item:
                if item['region_no'] == target_region_no:
                    return item['polygon']

    return None

def get_point_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        multi_points = loads(content)
        data_list = []
        # i = 0
        # while i < len(multi_point_geometry.geoms):
        for data in multi_points.geoms:
            # data_str = multi_point_geometry.geoms[i].wkt
            data_str = data.wkt
            coord_str = data_str[7:-2]
            x, y = coord_str.split()
            x = float(x)
            y = float(y)
            point_tuple = (x, y)
            data_list.append(point_tuple)
            # i += 1
        return data_list


# def read_wkt_from_file(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()
#     return content
#
# def parse_wkt_to_geometry(wkt_string):
#     return loads(wkt_string)


# def change_data_type(multi_point_geometry):
#     data_list = []
#     # i = 0
#     # while i < len(multi_point_geometry.geoms):
#     for data in multi_point_geometry.geoms:
#         # data_str = multi_point_geometry.geoms[i].wkt
#         data_str = data.wkt
#         coord_str = data_str[7:-2]
#         x, y = coord_str.split()
#         x = float(x)
#         y = float(y)
#         point_tuple = (x, y)
#         data_list.append(point_tuple)
#         # i += 1
#     return data_list

# def change_data_type(multi_point_geometry):
#     data_list = []
#     for data in multi_point_geometry.geoms:
#         data_str = data.wkt
#         coord_str = data_str[7:-1]  # Remove the 'POINT (' and ')' from the WKT
#         x, y = map(float, coord_str.split())
#         data_list.append(Point(x, y))  # Convert x, y to a Shapely Point object
#     return data_list


def get_point_density(data_list, polygon_coords, written_path, area):
    polygon = Polygon(polygon_coords)
    total_points = len(data_list)
    print(total_points)

    grid_points_count = {}
    min_x, min_y = polygon.bounds[0], polygon.bounds[1]
    max_x, max_y = polygon.bounds[2], polygon.bounds[3]

    for x, y in data_list:
        point = Point(x, y)
        if polygon.contains(point):
            # 计算该点所在小方格的索引
            grid_x = int((x - min_x) // area)
            grid_y = int((y - min_y) // area)

            # 更新该小方格内点的数量
            grid_points_count[(grid_x, grid_y)] = grid_points_count.get((grid_x, grid_y), 0) + 1

    with open(written_path, 'w') as f:
        f.write('x,y,density,normalized density\n')
        for grid_x in range(int(max_x - min_x) + 1):
            for grid_y in range(int(max_y - min_y) + 1):
                grid_density = grid_points_count.get((grid_x, grid_y), 0) / area  # 1为单位面积(1x1)
                grid_density_normalized = grid_points_count.get((grid_x, grid_y), 0) / (total_points)  # 归一化点密度
                # print(f"小方格 ({grid_x}, {grid_y}) 内的点密度为: {grid_density},归一化点密度为：{grid_density_normalized}")
                f.write(f"{grid_x}, {grid_y},{grid_density},{grid_density_normalized}\n")

def process_files_in_folder(parent_folder_path, json_file, output_folder_base, area,time_seperate):
    folder_list = os.listdir(parent_folder_path)

    for folder_name in folder_list:
        folder_path = os.path.join(parent_folder_path, folder_name, time_seperate)

        # Check if it is a folder
        if os.path.isdir(folder_path):
            region_no = folder_name[0:12]
            print(region_no)

            # Generate the output folder path based on the region_no
            output_folder = os.path.join(output_folder_base, folder_name, 'point_density',time_seperate)
            os.makedirs(output_folder, exist_ok=True)

        try:
            # Get the list of dictionaries containing "x" and "y" keys
            data_list = find_polygon_by_region_no(json_file, region_no)

            polygon_coords = []
            for point in data_list:
                polygon_coords.append((point["x"], point["y"]))

            # Process files in the current folder
            for file_name in os.listdir(folder_path):
                # Check if the file is a text file (you can modify this condition if needed)
                if file_name.endswith('.txt'):
                    # Form the full file path
                    file_path = os.path.join(folder_path, file_name)

                    # Generate the output file name based on the input file name
                    output_file_name = f"density_of_{os.path.splitext(file_name)[0]}.csv"
                    written_path = os.path.join(output_folder, output_file_name)

                    try:
                        # Read the WKT content from the file
                        # wkt_content = read_wkt_from_file(file_path)

                        # Parse WKT to Shapely geometry
                        # multi_point_geometry = parse_wkt_to_geometry(wkt_content)

                        # Convert WKT to a readable list of points
                        point_list = get_point_data(file_path)

                        # Calculate and output point density
                        get_point_density(point_list, polygon_coords, written_path, area)
                        print(f'Finish processing file: {file_name}')

                    except Exception as e:
                        # Print the error (you can handle it differently if needed)
                        print(f"An error occurred while processing {file_name}: {e}")
                        continue  # Skip this file and continue with the next one

            print(f'Finish processing folder: {folder_name}')

        except Exception as e:
            # Print the error (you can handle it differently if needed)
            print(f"An error occurred while processing folder {folder_name}: {e}")
            continue  # Skip this folder and continue with the next one


if __name__ == '__main__':
    parent_folder_path = 'D:/points/test_move'
    json_file = 'D:/points/whole_data/分组信息.json'
    output_folder_base = 'D:/points/test_move'
    area = 1
    # time_seperate = '0000-0700'
    time_seperate = '0700-0000'



    process_files_in_folder(parent_folder_path,json_file,output_folder_base, area, time_seperate)
    # folders_to_process = ['G32050700004gt1']  # Add the desired folder names here
    #
    # for folder_name_to_process in folders_to_process:
    #     folder_path = os.path.join(parent_folder_path, folder_name_to_process, time_seperate)
    #
    #     # Generate the output folder path based on the region_no
    #     output_folder = os.path.join(output_folder_base, folder_name_to_process, 'point_density')
    #     os.makedirs(output_folder, exist_ok=True)
    #
    #     try:
    #         # Get the list of dictionaries containing "x" and "y" keys
    #         data_list = find_polygon_by_region_no(json_file, folder_name_to_process[0:12])
    #
    #         polygon_coords = []
    #         for point in data_list:
    #             polygon_coords.append((point["x"], point["y"]))
    #
    #         # Process files in the specified folder
    #         for file_name in os.listdir(folder_path):
    #             # Check if the file is a text file (you can modify this condition if needed)
    #             if file_name.endswith('.txt'):
    #                 # Form the full file path
    #                 file_path = os.path.join(folder_path, file_name)
    #
    #                 # Generate the output file name based on the input file name
    #                 output_file_name = f"density_of_{os.path.splitext(file_name)[0]}.csv"
    #                 written_path = os.path.join(output_folder, output_file_name)
    #
    #                 try:
    #                     # Convert WKT to a readable list of points
    #                     point_list = get_point_data(file_path)
    #
    #                     # Calculate and output point density
    #                     get_point_density(point_list, polygon_coords, written_path, area)
    #                     print(f'Finish processing file: {file_name}')
    #
    #                 except Exception as e:
    #                     # Print the error (you can handle it differently if needed)
    #                     print(f"An error occurred while processing {file_name}: {e}")
    #                     continue  # Skip this file and continue with the next one
    #
    #         print(f'Finish processing folder: {folder_name_to_process}')
    #
    #     except Exception as e:
    #         # Print the error (you can handle it differently if needed)
    #         print(f"An error occurred while processing folder {folder_name_to_process}: {e}")
from shapely.wkt import loads
import csv

def get_point_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        multi_points = loads(content)
        x_coord = []
        y_coord = []
        # i = 0
        # while i < len(multi_point_geometry.geoms):
        for data in multi_points.geoms:
            # data_str = multi_point_geometry.geoms[i].wkt
            data_str = data.wkt
            coord_str = data_str[7:-2]
            x, y = coord_str.split()
            x = float(x)
            y = float(y)
            # point_tuple = (x, y)
            # data_list.append(point_tuple)
            # i += 1
            x_coord.append(x)
            y_coord.append(y)
        return x_coord, y_coord

if __name__ == '__main__':
    file_path = 'D:\\points\\test_with_speed\\G32050700004gt1\\202308031500-202308031530.txt'
    output_csv = "test_1500-1530.csv"
    x_coords, y_coords = get_point_data(file_path)
    with open(output_csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y"])  # Write header
        for x, y in zip(x_coords, y_coords):
            csv_writer.writerow([x, y])

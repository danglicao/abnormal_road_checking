# pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple paho-mqtt
# pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple protobuf
import paho.mqtt.client as mqtt
import proto.beautified_object_pb2 as dpe
from google.protobuf.json_format import MessageToDict
import schedule
import os
import numpy as np
from shapely.geometry import Point, Polygon
import json
from datetime import datetime
# from main import filtered_data
import cv2
import time as tm
from kafka import KafkaProducer
from kafka import Serializer
import logging
import csv
from datetime import datetime, time

# 连接回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(topic)  # 订阅需要的主题
    else:
        print("Failed to connect, return code: ", rc)

# 消息回调函数
def on_message(client, userdata, message):
    target_region_no = 'G32050700004'
    objs = dpe.BeautifiedObjects()
    objs.ParseFromString(message.payload)
    objs_dict = MessageToDict(objs)
    # print(objs)
    # for objs_dict in objs:
    obj_data = objs_dict['objects']
    # print(f'{obj_data}\n')
    # # print(type(obj_data))
    # print()
    for every_data in objs_dict['objects']:
        if 'velocity' in every_data and every_data['velocity'] >= 1 \
                and 'type' in every_data and every_data['type'] in [1, 4, 5, 6] \
                and 'groupNo' in every_data and every_data['groupNo'] == target_region_no:
            data_list.append({'position':every_data['position'],'timeMeas':every_data['timeMeas'], 'groupNo':every_data['groupNo']})
            # print(data_list)
    # print(type(objs_dict))
    # print(objs_dict)

    # print("Received message:", len(message.payload), "bytes")
    # print(type(objs))
    # print(f'type is{type(objs)}')
    # print(objs_dict.keys())
    # velocity = objs_dict['velocity']
    # position = objs_dict['position']
    #
    # with open(file_name, 'a') as f:
    #     f.write(f'{position}, {velocity}\n')
    # with open('test_of.json', 'w') as f:
    #     json.dump(objs_dict, f)

def output_data():
    current_time = tm.time()
    # print(current_time)
    print("Outputting data at", tm.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = int(1e6 * (current_time - 1800))  # 30分钟前的时间戳
    filtered_data = [data for data in data_list if int(data['timeMeas']) >= start_time]
    # print(filtered_data[0]['timeMeas'])
    print(filtered_data)
    return filtered_data

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT Broker")
    # 在此处添加重连逻辑
    while not client.is_connected():
        try:
            print("Attempting to reconnect...")
            tm.sleep(5)
            client.connect(broker_address, port=port)
        except:
            print("Reconnection failed. Retrying in 5 seconds...")
            tm.sleep(5)








def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def find_polygon_by_region_no(json_file, target_region_no):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'data' in data and isinstance(data['data'], list):
        for item in data['data']:
            if isinstance(item, dict) and 'region_no' in item and 'polygon' in item:
                if item['region_no'] == target_region_no:
                    return item['polygon']

    return None

def get_point_data(filtered_data):
    coordinates_list = []
    print(len(filtered_data))
    for data in filtered_data:
        if 'position' in data:
            position = data['position']
            x = position.get('x', 0)  # 默认值为0，你可以根据需求修改
            y = position.get('y', 0)  # 默认值为0，你可以根据需求修改
            coordinates_list.append((x, y))
    print(f'coord is {coordinates_list}')
    return coordinates_list

def is_time_between(start_time, end_time, check_time):
    if start_time <= end_time:
        return start_time <= check_time <= end_time
    else:
        return start_time <= check_time or check_time <= end_time

def get_normalized_density_array(data_list, polygon_coords, area):
    xy_coordinates = [(point['x'], point['y']) for point in polygon_coords]

    # 创建多边形对象
    polygon = Polygon(xy_coordinates)
    total_points = len(data_list)

    grid_points_count = {}
    min_x, min_y = polygon.bounds[0], polygon.bounds[1]
    max_x, max_y = polygon.bounds[2], polygon.bounds[3]

    for x, y in data_list:
        point = Point(x, y)
        if polygon.contains(point):
            grid_x = int((x - min_x) // area)
            grid_y = int((y - min_y) // area)
            grid_points_count[(grid_x, grid_y)] = grid_points_count.get((grid_x, grid_y), 0) + 1

    normalized_density_array = np.zeros((int(max_x - min_x) + 1, int(max_y - min_y) + 1))

    for grid_x in range(int(max_x - min_x) + 1):
        for grid_y in range(int(max_y - min_y) + 1):
            normalized_density_array[grid_x, grid_y] = grid_points_count.get((grid_x, grid_y), 0) / total_points

    return normalized_density_array

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


def get_absolute_coord(json_file, region_no):
    region_no = region_no[0:12]
    data_list = find_polygon_by_region_no(json_file, region_no)

    polygon_coords = []
    for point in data_list:
        polygon_coords.append((point["x"], point["y"]))
    polygon = Polygon(polygon_coords)
    min_x, min_y = polygon.bounds[0], polygon.bounds[1]
    max_x, max_y = polygon.bounds[2], polygon.bounds[3]
    return min_x, min_y, max_x, max_y

def get_coord(compare_array, region_json, region_no, compare_json, timestamp):
    min_x, min_y, max_x, max_y = get_absolute_coord(region_json, region_no)
    form_array = compare_json.get(region_no)
    result_array = np.where(compare_array < form_array, 255, compare_array)
    result_array = np.rot90(result_array, k=1)
    # 应用膨胀腐蚀
    dilated_array = dilation_erosion(result_array.astype(np.uint8))

    dilated_filterd_array = connect_component(dilated_array)

    contours, hierarchy = cv2.findContours(dilated_filterd_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_list = []
    # wkt_dict = {'timestamp': timestamp}
    wkt_dict = {}
    for contour in contours:
        contour_list = []
        for point in contour:
            # print(type(point))
            x, y = point[0]  # 提取每个点的坐标 (x, y)
            print(f"Point: ({x + min_x}, {max_y - y})")
            contour_list.append((x + min_x, max_y - y))
        print('a')
        contours_list.append(contour_list)
    wkt_list = []
    for idx, contour_position in enumerate(contours_list, start=1):
        # 将每个轮廓的 numpy 数组转换为 WKT 格式
        if len(contour_position) >= 3:
            # 获得每个轮廓的第一个点
            first_point = contour_position[0]
            wkt_coordinates = "POLYGON ((" + ", ".join(
                [f"{x} {y}" for x, y in contour_position]) + f", {first_point[0]} {first_point[1]}))"
            # wkt_dict[f'polygon{idx}'] = wkt_coordinates
            # wkt_dict['polygon'] = wkt_coordinates
            wkt_list.append(wkt_coordinates)

    return wkt_list


def write_down_base(target_region_no, json_file, output_folder, area):
    timestamp = tm.time()
    timestamp = int(timestamp * 1e6)  # 获取当前时间戳
    global filtered_data

    polygon = find_polygon_by_region_no(json_file, target_region_no)
    point_list = get_point_data(filtered_data)
    density_array = get_normalized_density_array(point_list, polygon, area)
    rows, cols = density_array.shape
    current_time = datetime.now().time()
    start_time = time(7, 29, 59)  # 7:00 AM
    end_time = time(0, 29, 59)  # 11:59 PM

    if is_time_between(start_time, end_time, current_time):
        sub_folder = '0700-0000'
    else:
        sub_folder = '0000-0700'
    output_full_folder = os.path.join(output_folder, sub_folder)
    if not os.path.exists(output_full_folder):
        os.makedirs(output_full_folder)
    file_path = os.path.join(output_full_folder, f'{timestamp}' + ".csv")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, f'{timestamp}' + ".csv")
    with open(file_path, mode='w', newline='') as csv_file:
        fieldnames = ['x', 'y', 'normalized density']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()  # 写入 CSV 文件头部

        for x in range(rows):
            for y in range(cols):
                writer.writerow({
                    'x': x,
                    'y': y,
                    'normalized density': density_array[x, y]
                })

    print("CSV file has been created:", file_path)

def output_task(target_region_no, json_file, output_folder, compare_json):
    timestamp = tm.time()
    timestamp = int(timestamp * 1e6)# 获取当前时间戳
    global filtered_data
    filtered_data = output_data()
    current_time = datetime.now().time()
    start_time = time(7, 29, 59)  # 7:00 AM
    end_time = time(0, 29, 59)  # 11:59 PM

    if is_time_between(start_time, end_time, current_time):
        json_data = read_json_file(compare_json)

        polygon = find_polygon_by_region_no(json_file, target_region_no)
        point_list = get_point_data(filtered_data)
        density_array = get_normalized_density_array(point_list, polygon, area)
    else:
        density_array = np.array([])
    wkt_list = get_coord(density_array, json_file, target_region_no,json_data, timestamp)

    # 打印或使用 coords
    # print(coords)

    print(f'coords is {wkt_list}\n')
    print('test')
    # file_path = os.path.join(output_folder, f'{timestamp}' + ".txt")
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # with open(file_path, "w") as file:
    #     file.write(wkt_coordinates)
    # 将数据转换为 JSON 格式
    i = 0
    for value in wkt_list:
        message = {'timestamp': timestamp, 'polygon':value}
        json_message = json.dumps(message)

        # 将 JSON 数据发送到 Kafka 主题
        kafka_producer.send(topic_kafka, value=json_message.encode('utf-8'))
        i += 1
    kafka_producer.flush()
    print(i)
    print("Outputting filtered_data:")
    print(len(filtered_data))
    # for data in filtered_data:
        # print(data)
    print("========================================")

def on_log(client, userdata, level, buf):
    print(f"MQTT log: {buf}")

if __name__ == '__main__':
    target_region_no = 'G32050700004' # 道路编号
    json_file = 'D:/points/danglic/server_version/information_of_group.json' #本目录下的information_of_group文件，包含
    area = 1 # 切割的放个面积
    output_folder = 'D:/points/danglic/server_version/G32050700004' # 输出的存档文件文件夹
    compare_json = 'D:/points/danglic/server_version/arrays_data_test_1.5_multipler.json' # 用存档文件生成的可用于比较密度差异的json文件
    # 配置日志记录
    logging.basicConfig(level=logging.DEBUG,  # 设置日志级别为DEBUG，记录所有级别的日志
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        # filename='app.log',  # 指定日志文件名
                        # filemode='w')  # 设置文件写入模式为覆盖写入
                        )

    filtered_data = []

    # 设置 MQTT 服务器地址和端口
    broker_address = "192.168.9.4"
    port = 1883
    user = "allride"
    passwd = "T3hwQTGNPPvYV&8u"
    topic = "road/dpe"
    file_name = 'test_data.txt'
    data_list = []

    # 设置 Kafka 服务器地址和端口
    kafka_broker = "192.168.2.24:9092"  # 替换为实际的 Kafka 服务器地址

    topic_kafka = 'abnormal_road_checking'

    # 配置 Kafka 生产者
    producer_config = {
        "bootstrap_servers": kafka_broker
    }
    # 创建kafka生产者实例
    kafka_producer = KafkaProducer(**producer_config)
    # try:
    #     # 创建 Kafka 生产者实例
    #     producer = KafkaProducer(bootstrap_servers=kafka_broker)
    #
    #     # 判断是否成功创建实例
    #     if producer.bootstrap_connected():
    #         print("Kafka producer instance created successfully!")
    #         # 定义一个日志记录器
    #         logger = logging.getLogger(__name__)
    #
    #         # 打印不同级别的日志
    #         logger.debug('This is a debug message')
    #         logger.info('This is an info message')
    #         logger.warning('This is a warning message')
    #         logger.error('This is an error message')
    #         logger.critical('This is a critical message')
    #
    #     else:
    #         print("Failed to create Kafka producer instance.")
    #
    #     # 关闭 Kafka 生产者连接
    #     # producer.close()
    # except Exception as e:
    #     print("An error occurred:", str(e))

    # 创建 MQTT 客户端
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # 连接到 MQTT 服务器
    client.username_pw_set(user, passwd)
    client.connect(broker_address, port=port)

    # 每隔5分钟执行一次输出任务
    custom_output_task = output_task(target_region_no, json_file,output_folder, compare_json)
    custom_write_down_base = write_down_base(target_region_no, json_file, output_folder, area)
    schedule.every(30).minutes.do(custom_output_task)
    schedule.every(30).minutes.do(custom_write_down_base)

    # ...其他代码...

    # 开始循环，处理消息
    client.loop_start()
    # client.on_log = on_log

    client.loop_start()

    try:
        while True:
            schedule.run_pending()  # 执行定时任务
            tm.sleep(1)
            tm.sleep(1)
    except KeyboardInterrupt:
        # Ctrl+C 停止循环并断开连接
        client.loop_stop()
        client.disconnect()
        print("Code stopped manually")
    print(filtered_data)
    # # 调用函数进行处理
    # process_filtered_data(filtered_data)







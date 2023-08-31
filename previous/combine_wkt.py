from shapely.geometry import MultiPoint
import os
from shapely.wkt import loads

def merge_wkt_files_in_batches(folder_path, output_folder):
    # 获取文件夹中所有的文件名
    file_names = os.listdir(folder_path)

    # 每3个文件一组进行合并
    batch_size = 3
    num_files = len(file_names)
    num_batches = num_files // batch_size

    for batch_idx in range(num_batches):
        # 获取当前批次的文件名列表
        batch_files = file_names[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # 创建一个空的Multipoint对象，用于存储当前批次的所有点
        merged_multipoint = MultiPoint()

        # 遍历当前批次的文件，读取WKT格式的Multipoint数据并合并
        for file_name in batch_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                wkt_data = file.read()
                multipoint = loads(wkt_data)
                merged_multipoint = merged_multipoint.union(multipoint)

        # 将合并后的Multipoint对象转换为WKT格式的字符串
        merged_wkt = merged_multipoint.wkt

        # 生成合并后的WKT文件名（可以根据需要自定义命名规则）
        output_file_name = f"batch_{batch_idx + 1}.txt"

        # 将WKT字符串保存到新文件中
        output_file_path = os.path.join(output_folder, output_file_name)
        with open(output_file_path, 'w') as file:
            file.write(merged_wkt)

# 示例用法
# 替换为你的文件夹路径和输出文件夹路径
folder_path = 'D:\points\whole_data\data'
output_folder = 'D:\points\whole_data\combine_data'
merge_wkt_files_in_batches(folder_path, output_folder)

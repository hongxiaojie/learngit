"""
step1. 转换数据集
step2. 数据集分类
step3. 划分数据集
"""
import os
import random
import json
import shutil
import cv2
from prettytable import PrettyTable


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def position(pos, w, h):
    """获取xmin,ymin,xmax,ymax"""
    x = []
    y = []
    nums = len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max = min(max(x), w - 1)   # 避免超出边界
    x_min = max(min(x), 0)
    y_max = min(max(y), h - 1)
    y_min = max(min(y), 0)
    b = (float(x_min), float(x_max), float(y_min), float(y_max))
    return b


def convert(size, box):
    """将xmin,ymin,xmax,ymax转为x,y,w,h中心点坐标和宽高，并归一化"""
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def de_convert(xywh, size=[513, 513]):
    x, y, w, h = xywh
    h_s, w_s = size
    x_new = x * w_s
    y_new = y * h_s
    w_new = w * w_s
    h_new = h * h_s
    x1 = x_new - w_new // 2
    x2 = x_new + w_new // 2
    y1 = y_new - h_new // 2
    y2 = y_new + h_new // 2

    return int(x1), int(y1), int(x2), int(y2)


def get_class_name(input_path):
    json_files = [f for f in os.listdir(input_path) if f.endswith(".json")]

    labels = {}
    print("---> start.")
    for file_idx, file in enumerate(json_files):
        json_path = os.path.join(input_path, file)
        with open(json_path, "r", encoding='UTF-8') as f:
            json_data = json.load(f)

        for idx, shape in enumerate(json_data["shapes"]):
            label_name = shape['label']

            if label_name == 'bui':
                print("bui:", json_path)

            if label_name not in labels:
                labels[label_name] = 0
            labels[label_name] += 1
                # labels.append(label_name)

    print("\t处理完{}个文件".format(file_idx + 1))
    print("\t包含类别：{}".format(labels))
    print("---> done.")


def convert_annotation(dir_path, output_dir):
    json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

    # output_dir = os.path.join(dir_path, "txt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("输出文件夹不存在，创建输出文件夹{}".format(output_dir))

    empty_json_num = 0
    for file_idx, file in enumerate(json_files):
        json_path = os.path.join(dir_path, file)
        with open(json_path, "r", encoding='UTF-8') as f:
            json_data = json.load(f)
        h = json_data['imageHeight']  # 获取原图的宽高，用于归一化
        w = json_data['imageWidth']

        if not json_data["shapes"]:
            # print(file)
            empty_json_num += 1

        file_name = os.path.splitext(file)[0] + ".txt"

        with open(os.path.join(output_dir, file_name), "w") as f:
            for idx, shape in enumerate(json_data["shapes"]):
                label_name = shape['label']
                points = shape['points']
                xxyy = position(points, w, h)
                xywh = convert((w, h), xxyy)
                if xywh[0] > 1 or xywh[1] > 1 or xywh[2] > 1 or xywh[3] > 1:
                    print("\t{}: out of bounds coordinate labels.".format(json_path))
                # class_dict = {"kno": 0, "deg": 1, "pro": 2, "sho1": 3, "gap": 4}
                # class_dict = {"kno": 0, "deg": 1, "pro": 2, "sho1": 3, "gap": 4, "sho": 5, "deg1": 6, "scr": 7, "bul": 8,
                #               "degd": 9, "nbl": 10, "lac": 11, "low": 12, "cra": 13, "spi": 14, "high": 15}
                # class_dict = {"kno": 0, "deg": 1, "pro": 2, "sho1": 3, "gap": 4, "sho": 5, "deg1": 6, "scr": 7, "bul": 8,
                #               "degd": 9, "nbl": 10, "lac": 11, "low": 12, "spi": 13, "high": 14, "pro1": 15}
                # class_dict = {"kno": 0, "deg": 1, "pro": 2, "sho1": 3, "gap": 4, "sho": 5, "deg1": 6, "scr": 7, "bul": 8,
                #               "degd": 9, "nbl": 10, "lgap": 11, "low": 12, "spi": 13, "high": 14, "prom": 15}
                # class_dict = {"kno": 0, "deg": 1, "pro": 2, "sho1": 3, "gap": 4, "sho": 5, "deg1": 6, "scr": 7,
                #               "bul": 8, "degd": 9, "nbl": 10, "lgap": 11, "spi": 12, "prom": 13, "spiw": 14}

                # class_dict = {'qb': 0, 'bj': 1, 'pz': 2, 'bx': 3, 'ps': 4, 'jm': 5, 'bq': 6, 'ps1': 7}
                class_dict = {'td': 0, 'xt': 1, 'hd': 2, 'ls': 3, 'sz': 4, 'fp': 5}
                # class_dict = {'xt': 0, 'hd': 1, 'ls': 2, 'sz': 3, 'fp': 4}
                if label_name == 'cra':
                    print("cra: ", file)
                    continue

                if label_name == 'lac':
                    print("lac: ", file)
                    continue

                if label_name == 'imp':
                    print("imp: ", file)
                    continue

                if label_name == 'low':
                    continue

                if label_name == 'high':
                    continue

                if label_name != 'td':
                    continue

                # if label_name == 'td':
                #     continue

                cls_id = class_dict[label_name]

                # file_name = os.path.splitext(file)[0] + ".txt"
                # with open(os.path.join(output_dir, file_name), "w") as f:
                #     f.write(str(cls_id) + " " + " ".join([str(p) for p in xywh]) + '\n')
                f.write(str(cls_id) + " " + " ".join([str(p) for p in xywh]) + '\n')

    print("\t处理完{}个文件, 空json数量：{}".format(file_idx + 1, empty_json_num))


def visualize(txt_dir, img_dir, output_dir):

    txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
    # output_dir = os.path.join(txt_dir, "txt_vis")

    print('--> start.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("输出文件夹不存在，创建输出文件夹{}".format(output_dir))

    for idx, txt in enumerate(txt_files):
        if idx > 250:
            print("done.the number of files exceeds 100, only the first 100 are saved")
            break
        txt_path = os.path.join(txt_dir, txt)
        with open(txt_path, "r") as f:
            lines = f.readlines()

        img_name = os.path.splitext(txt)[0] + ".bmp"
        # img_name = os.path.splitext(txt)[0] + ".png"
        img = cv2.imread(os.path.join(img_dir, img_name))

        for item in lines:
            data = [float(d) for d in item.split(' ')]
            label, x_c, y_c, w, h = data

            x1, y1, x2, y2 = de_convert((x_c, y_c, w, h), size=img.shape[:2])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, str(int(label)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_dir, img_name), img)


def split_class(data_dir, target_dir):
    print("--> split_class start")
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    class_count = {}

    for file_idx, file in enumerate(json_files):
        try:
            json_path = os.path.join(data_dir, file)
            with open(json_path, "r", encoding='UTF-8') as f:
                json_data = json.load(f)

            for idx, shape in enumerate(json_data["shapes"]):
                label_name = shape['label']
                class_dir = os.path.join(target_dir, label_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                if idx == 0:
                    # img_name = os.path.splitext(file)[0] + ".png"
                    img_name = os.path.splitext(file)[0] + ".bmp"
                    txt_name = os.path.splitext(file)[0] + ".txt"
                    img_src_path = os.path.join(data_dir, img_name)
                    txt_src_path = os.path.join(data_dir, txt_name)
                    # img_dis_path = os.path.join(class_dir, txt_name)
                    # txt_dis_path = os.path.join(class_dir, img_name)
                    # shutil.move(img_src_path, img_dis_path)
                    # shutil.move(txt_src_path, txt_dis_path)
                    shutil.copy2(img_src_path, class_dir)
                    shutil.copy2(txt_src_path, class_dir)
                    shutil.copy2(json_path, class_dir)

                    if label_name not in class_count:
                        class_count[label_name] = 1
                    else:
                        class_count[label_name] += 1
                else:
                    break
        except FileNotFoundError:
            print(file)
            continue

    tb = PrettyTable()
    tb.field_names = ["class name", ] + list(class_count.keys())
    tb.add_row([file_idx + 1, ] + list(class_count.values()))
    tb.align = 'l'
    print(tb)


def split_dataset(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    """
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    """
    class_names = [f for f in os.listdir(src_data_folder) if os.path.isdir(os.path.join(src_data_folder, f))]
    # 在目标目录下创建文件夹
    for atr in ['images', 'labels']:
        for split_name in ['train', 'val', 'test']:
            split_path = os.path.join(target_data_folder, atr, split_name)
            if not os.path.exists(split_path):
                os.makedirs(split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    random.seed(0)

    tb = PrettyTable()
    tb.add_column("class name", ["all", "train", "val", "test"])

    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = [f for f in os.listdir(current_class_data_path) if f.endswith(".txt")]
        # current_all_data = [f for f in os.listdir(current_class_data_path) if f.endswith(".json")]
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_images_folder = os.path.join(target_data_folder, 'images', 'train')
        train_labels_folder = os.path.join(target_data_folder, 'labels', 'train')
        val_images_folder = os.path.join(target_data_folder, 'images', 'val')
        val_labels_folder = os.path.join(target_data_folder, 'labels', 'val')
        test_images_folder = os.path.join(target_data_folder, 'images', 'test')
        test_labels_folder = os.path.join(target_data_folder, 'labels', 'test')

        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        # test_stop_flag = math.floor(current_data_length * test_scale)
        # val_stop_flag = current_data_length - test_stop_flag
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_txt_path = os.path.join(current_class_data_path, current_all_data[i])
            src_img_path = os.path.splitext(src_txt_path)[0] + ".bmp"
            # src_img_path = os.path.splitext(src_txt_path)[0] + ".png"
            if current_idx <= train_stop_flag:
                shutil.copy2(src_txt_path, train_labels_folder)
                shutil.copy2(src_img_path, train_images_folder)
                # print("{}复制到了{}".format(src_txt_path, train_folder))
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                shutil.copy2(src_txt_path, val_labels_folder)
                shutil.copy2(src_img_path, val_images_folder)
                # print("{}复制到了{}".format(src_txt_path, val_folder))
                val_num = val_num + 1
            else:
                shutil.copy2(src_txt_path, test_labels_folder)
                shutil.copy2(src_img_path, test_images_folder)
                # print("{}复制到了{}".format(src_txt_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1

        tb.add_column(class_name, [current_data_length, train_num, val_num, test_num])
    tb.align = 'l'
    print(tb)


if __name__ == '__main__':

    # root_dir = r"..\..\yolov5-datasets\Cigarette\dataset_all"
    # root_dir = r"F:\NWT\Harry_DE\Cigarette-1203\dataset_all"
    # root_dir = r"F:\NWT\Harry_DE\dataset_all(1)"
    # root_dir = r"F:\NWT\Harry_DE\Cigarrete"
    # root_dir = r"E:\my_project\yolov5-datasets\potato_location\dataset_all"
    # root_dir = r"E:\my_project\yolov5-datasets\potato_location_dw\dataset_all"
    # get_class_name(root_dir)

    txt_dir = r"E:\my_project\yolov5-datasets\potato_location_1223\dataset_all"
    img_dir = r"E:\my_project\yolov5-datasets\potato_location_1223\dataset_all"
    output_dir = r"E:\my_project\yolov5-datasets\potato_location_1223\vis"
    visualize(txt_dir, img_dir, output_dir)
    # #
    train_scale, val_scale, test_scale = 0.9, 0.05, 0.05

    # root_dir = "../datasets/Cigarette"
    # root_dir = r"..\..\yolov5-datasets\D1022"
    # root_dir = r"E:\my_project\yolov5-datasets\potato_defect"
    # root_dir = r"E:\my_project\yolov5-datasets\potato_location_1223"
    #
    # json_dir = os.path.join(root_dir, "dataset_all")
    # txt_dir = json_dir
    # class_dir = os.path.join(root_dir, "classed")
    #
    # print(colorstr(f"训练集：验证集：测试集 = {train_scale}：{val_scale}：{test_scale}"))
    #
    # # step1. 转换数据集
    # print(colorstr("---> step1 start .."))
    # convert_annotation(json_dir, txt_dir)
    # print(colorstr("---> step1 done.\n"))
    #
    # # step2. 数据集分类
    # print(colorstr("---> step2 start .."))
    # split_class(json_dir, class_dir)
    # print(colorstr("---> step2 done.\n"))
    #
    # # step3. 划分数据集
    # print(colorstr("---> step3 start .."))
    # split_dataset(class_dir, root_dir, train_scale=train_scale, val_scale=val_scale, test_scale=test_scale)
    # print(colorstr("---> step3 done."))


import os
import torch
from PIL import Image

kws = ['工业机器人', '数控机床', '数控系统', '书籍', '图表', '人类']
# images_path_open 参数 二选一注释掉！
# images_path_open = os.path.join(os.getcwd(), "industry_data", "industry_photos")    # 重命名训练集
images_path_open = os.path.join(os.getcwd(), "test_images")    # 重命名测试集

for kw in kws:
    # 重命名图片目录
    image_path = os.path.join(images_path_open, kw)
    # 获取图片列表（图片名.后缀名）
    img_list = os.listdir(image_path)
    count = len(img_list)
    print("-------------总计{}张图片, {}重命名开始--------------".format(count, kw))
    num = 0
    # 图片目的保存目录 images_path_open\rename_images
    image_path_save_dir = os.path.join(images_path_open, "rename_images")
    if not os.path.exists(image_path_save_dir):
        os.mkdir(image_path_save_dir)
    # 逐级生成图片目的保存目录目录 images_path_open\rename_images\kw
    image_path_save_dir = os.path.join(image_path_save_dir, kw)
    if not os.path.exists(image_path_save_dir):
        os.mkdir(image_path_save_dir)
    # 遍历图片列表
    for img in img_list:
        # 图片保存全路径
        image_path_name = os.path.join(image_path, img)
        # 加载图片
        image_open = Image.open(image_path_name).convert('RGB')
        name = kw + "_" + str(num) + '.png'  # 根据关键字和序号命名
        image_path_save_fullname = os.path.join(image_path_save_dir, name)
        filenames = image_path_save_fullname.split('\\')
        lens = len(filenames)
        print("当前图片：" + filenames[lens-1])
        # 保存图片
        image_open.save(image_path_save_fullname, 'png')
        num += 1
    print("-------------总计{}张图片, {}重命名结束--------------".format(count, kw))

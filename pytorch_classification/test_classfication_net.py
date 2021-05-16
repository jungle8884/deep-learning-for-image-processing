import os
import json
import torch
from PIL import Image
from torchvision import transforms
from pytorch_classification.Test3_vggnet.model import vgg
from pytorch_classification.Test5_resnet.model import resnet50
from matplotlib import font_manager as fm, rcParams, pyplot as plt

# 全局变量
major_url = 'https://image.baidu.com/search/index?'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/84.0.4147.135 Safari/537.36'}
kws = ['工业机器人', '数控机床', '数控系统', '书籍', '图表', '人类']
data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
path_data_set = os.path.join(data_root, "data_set")
num_class = 6

# 统计post_class_fication-当前类别的总数目
post_class_fication_count = {}
# 统计post_class_fication-当前类别的分类错误数目
post_class_fication_false_count = {}

# 权重参数路径
# weights_path = os.path.join(os.getcwd(), "Test3_vggnet", "vgg16Net.pth")
weights_path = os.path.join(os.getcwd(), "Test5_resnet", "resNet50.pth")
# 类别路径
# json_path = os.path.join(os.getcwd(), "Test3_vggnet", "class_indices.json")
json_path = os.path.join(os.getcwd(), "Test5_resnet", "class_indices.json")


# 分类图片
def predictImage_vgg(weights_path, json_path, img="../1.jpeg"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = img
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        img_names = img_path.split('\\')
        lens = len(img_names)
        img_name = img_names[lens - 1]
        # print(e)
        print("{} 无法打开!".format(img_name))
        return '未知', 0
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model， 记得修改num_classes=类别数
    model = vgg(model_name="vgg16", num_classes=num_class).to(device)
    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # argmax最大值对应的索引，numpy再转为ndarray变量
    class_fication = class_indict[str(predict_cla)]
    # print(class_fication)
    prob = predict[predict_cla].numpy()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    # plt.title(print_res)
    # plt.show()
    return class_fication, prob


# resnet分类图片
def predictImage_resnet(weights_path, json_path, img="../1.jpeg"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("use {}".format(device))

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = img
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        img_names = img_path.split('\\')
        lens = len(img_names)
        img_name = img_names[lens - 1]
        # print(e)
        print("{} 无法打开!".format(img_name))
        return '未知', 0
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model  需要修改 默认num_classes=5
    model = resnet50(num_classes=num_class).to(device)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # argmax最大值对应的索引，numpy再转为ndarray变量
    class_fication = class_indict[str(predict_cla)]
    # print(class_fication)
    prob = predict[predict_cla].numpy()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    # plt.title(print_res)
    # plt.show()
    return class_fication, prob


# 对图片进行分类
def classfication(classfication_name):
    # 是否保存分类后的图片
    # is_save = False  # 默认不保存
    is_save = True
    # 初始化统计参数
    for kw in kws:
        post_class_fication_count[kw] = 0
        post_class_fication_false_count[kw] = 0
    # 分类处理图片
    for kw in kws:
        # 图片目录
        image_path = os.path.join(path_data_set, "test_images", kw)
        # 获取图片列表（图片名.后缀名）
        img_list = os.listdir(image_path)
        count = len(img_list)
        print("-------------总计{}张图片, {}分类开始--------------".format(count, kw))
        # 遍历图片列表
        for img in img_list:
            # 获得图片全路径
            image_path_name = os.path.join(image_path, img)
            try:
                # 设置预测函数：
                post_class_fication, prob = predictImage_vgg(weights_path, json_path, image_path_name)
                # post_class_fication, prob = predictImage_resnet(weights_path, json_path, image_path_name)
                if prob > 0.9:
                    # 显示图片分类结果
                    print_res = "class: {}   prob: {:.3}".format(post_class_fication, prob)
                    print("\n" + img + " 的图片分类结果: \n\t" + print_res)
                    # 统计post_class_fication-当前类别的数目
                    post_class_fication_count[post_class_fication] += 1
                    # 统计post_class_fication-当前类别的分类错误数目
                    if post_class_fication != str(img.split('_')[0]):
                        print("\t分类错误!")
                        post_class_fication_false_count[post_class_fication] += 1
                    if is_save:
                        # 加载图片
                        image_open = Image.open(image_path_name).convert('RGB')
                        # 图片目的保存目录 path_data_set\classfication_name\post_class_fication\
                        image_dir_destination = os.path.join(path_data_set,
                                                             classfication_name)
                        if not os.path.exists(image_dir_destination):
                            os.mkdir(image_dir_destination)
                        # 逐级生成目录
                        image_dir_destination = os.path.join(image_dir_destination,
                                                             str(post_class_fication))
                        if not os.path.exists(image_dir_destination):
                            os.mkdir(image_dir_destination)
                        # 图片目的保存文件名全路径
                        image_save_destination = os.path.join(image_dir_destination, str(img))
                        # 截取全路径的 class\name 类别名称\图片名
                        class_name = image_save_destination.split('\\')
                        lens = len(class_name)
                        print("分类结果图片保存路径为: {}\\{}\n".format(class_name[lens - 2], class_name[lens - 1]))
                        # 保存图片，不需要保存图片则注释掉！
                        image_open.save(image_save_destination, 'png')
            except Exception as e:
                print(e)
                continue
        print("-------------总计{}张图片, {}分类结束--------------".format(count, kw))
    # 统计准确度：
    for kw in kws:
        print("\n统计{}-当前类别的数目：{}".format(kw, post_class_fication_count[kw]))
        print("统计{}-当前类别的分类错误数目：{}".format(kw, post_class_fication_false_count[kw]))
        # 保留2位有效数字
        error_rate = float("{0:.2f}".format(post_class_fication_false_count[kw] / post_class_fication_count[kw]))
        accuracy = (1 - error_rate)
        print("{}-当前类别的分类准确率：{}\n".format(kw, accuracy))


# 根据下载的图片进行分类，并且保存到数据集目录下
if __name__ == '__main__':
    # net_name = "vgg16"
    net_name = "resnet50"
    classfication(net_name + "_test_result_images")

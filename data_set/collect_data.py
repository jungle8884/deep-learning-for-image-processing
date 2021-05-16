import requests
import urllib.parse as up
import time
import os

from pytorch_classification.test_classfication_net import predictImage_resnet, predictImage_vgg


# 全局变量
major_url = 'https://image.baidu.com/search/index?'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/84.0.4147.135 Safari/537.36'}
# 一共多少类: ['工业机器人', '数控机床', '数控系统', '书籍', '图表', '人类']
# 定义人类类别 --> '人类': '车间工人', '车间领导', '车间员工'
kws_human = tuple(['车间工人', '车间领导', '车间员工'])
# 下载关键字
# kws = ['工业机器人', '数控机床', '数控系统', '书籍', '图表', '车间工人', '车间领导', '车间员工']
# kws = ['工业机器人', '车间工人', '车间领导', '车间员工']  # 只下载人类图片
kws = ['工业机器人']


# 权重参数路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
# weights_path = os.path.join(data_root, "pytorch_classification", "Test3_vggnet", "vgg16Net.pth")
weights_path = os.path.join(data_root, "pytorch_classification", "Test5_resnet", "resNet50.pth")
# 类别路径
# json_path = os.path.join(data_root, "pytorch_classification", "Test3_vggnet", "class_indices.json")
json_path = os.path.join(data_root, "pytorch_classification", "Test5_resnet", "class_indices.json")


# 下载page=end_page-start_page多少页数据，每页30张图片，需要自己更改page参数！
def pic_spider(keyword="keyword",
               start_page=0, end_page=10,
               file_path="temp_images",
               is_classfication=False):
    # 下载计数
    num_count = 0
    # 分级判断
    path = os.path.join(os.getcwd(), file_path)
    if not os.path.exists(path):
        os.mkdir(path)
    # 如果关键字属于人类的话则归为人类
    if keyword in kws_human:
        path = os.path.join(path, "人类")
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        path = os.path.join(path, keyword)
        if not os.path.exists(path):
            os.mkdir(path)
    if keyword != '':
        for num in range(end_page):
            # 设置起始页
            if num < start_page:
                num_count = num_count + 30
                continue
            # 比较几个请求传递参数后，只有 queryWord, word, pn, gsm 传递的参数变化
            # queryWord 和 word 都是要搜索的关键词
            # pn 图片数量
            # gsm 图片数量所对应的八进制
            data = {
                "tn": "resultjson_com",
                "logid": "11587207680030063767",
                "ipn": "rj",
                "ct": "201326592",
                "is": "",
                "fp": "result",
                "queryWord": keyword,
                "cl": "2",
                "lm": "-1",
                "ie": "utf-8",
                "oe": "utf-8",
                "adpicid": "",
                "st": "-1",
                "z": "",
                "ic": "0",
                "hd": "",
                "latest": "",
                "copyright": "",
                "word": keyword,
                "s": "",
                "se": "",
                "tab": "",
                "width": "",
                "height": "",
                "face": "0",
                "istype": "2",
                "qc": "",
                "nc": "1",
                "fr": "",
                "expermode": "",
                "force": "",
                "pn": num * 30,
                "rn": "30",
                "gsm": oct(num * 30),
                "1602481599433": ""
            }
            url = major_url + up.urlencode(data)
            i = 10
            pic_list = []
            while i > 0:
                try:
                    pic_list = requests.get(url=url, headers=headers).json().get('data')
                    break
                except Exception as e:
                    print(e)
                    print('{} 网络不好，正在重试...'.format(i))
                    i -= 1
                    time.sleep(1.3)
            for pic in pic_list:
                url = pic.get('thumbURL', '')  # 有的没有图片链接，就设置成空
                if url == '':
                    continue
                # name = pic.get('fromPageTitleEnc')
                # for char in ['?', '\\', '/', '*', '"', '|', ':', '<', '>']:
                #     name = name.replace(char, '')   # 将所有不能出现在文件名中的字符去除掉
                name = str(keyword) + '_' + str(num_count)
                type_pic = pic.get('type', 'png')  # 找到图片的类型，若没有找到，默认为 png
                pic_path = (os.path.join(path, '%s.%s') % (name, type_pic))
                if not os.path.exists(pic_path):
                    with open(pic_path, 'wb') as f:
                        img = requests.get(url=url, headers=headers).content
                        # 下载时是否使用分类来判断
                        if is_classfication:
                            # 先下载下来
                            f.write(img)
                            post_class_fication, prob = predictImage_resnet(weights_path, json_path, pic_path)
                            # post_class_fication, prob = predictImage_vgg(weights_path, json_path, pic_path)
                            # 满足条件则 计数器加1
                            # if prob > 0.9:
                            if prob > 0.9 and post_class_fication == "人类":  # 只下载工业机器人中被误判为：类别属于'人类'的图片
                                print("\n{}: {}-{} 已完成下载\n".format(num_count, post_class_fication, name))
                            # 否则, 删除刚刚下载的图片
                            else:
                                print("下载的图片不符合要求, 删除: {}".format(name))
                                f.close()
                                try:
                                    os.remove(pic_path)
                                except Exception as e:
                                    # print(e)
                                    print("无法删除图片不符合要求的图片: {}".format(name))
                            num_count += 1
                        # 不用判断直接下载
                        else:
                            f.write(img)
                            print(name, '已完成下载')
                            num_count += 1


# 下载训练数据图片
def download_train():
    for kw in kws:
        # 使用爬虫下载图片
        # pic_spider(kw, start_page=11, end_page=60, file_path="train_images")  # train images
        pic_spider(kw,
                   start_page=0, end_page=2000,
                   file_path="train_images",
                   is_classfication=True)  # train images


# 下载测试数据图片
def download_test():
    for kw in kws:
        # 使用爬虫下载图片
        # pic_spider(kw, start_page=0, end_page=10, file_path="test_images")  # test images
        pic_spider(kw,
                   start_page=0, end_page=10,
                   file_path="test_images",
                   is_classfication=True)  # test images


if __name__ == '__main__':
    download_train()
    # download_test()

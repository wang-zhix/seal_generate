import os
import cv2
import numpy as np
from math import pi, cos, sin, tan
from random import randint, choices
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from multiprocessing import Process, Queue
import random

# 获取文件夹下文件列表
def get_files_list(path):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    return files

# 判断字符是否为中文
def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False

# 计算五角星各个顶点
# int R:五角星的长轴
# int x, y:五角星的中心点
# int yDegree:长轴与y轴的夹角
def pentagram(x, y, R, yDegree=0):
    rad = pi / 180  # 每度的弧度值
    r = R * sin(18 * rad) / cos(36 * rad)  # 五角星短轴的长度

    # 求取外圈点坐标
    RVertex = [(x - (R * cos((90 + k * 72 + yDegree) * rad)), y - (R * sin((90 + k * 72 + yDegree) * rad))) for k in
               range(5)]
    # 求取内圈点坐标
    rVertex = [(x - (r * cos((90 + 36 + k * 72 + yDegree) * rad)), y - (r * sin((90 + 36 + k * 72 + yDegree) * rad)))
               for k in range(5)]

    # 顶点左边交叉合并
    vertex = [x for y in zip(RVertex, rVertex) for x in y]
    return vertex

# 计算圆的上下左右切点
def circle(x, y, r):
    return (x - r, y - r, x + r, y + r)

class Stamp:
    def __init__(self, edge=5,  # 图片边缘空白的距离
                 H=160,  # 圆心到中层文字下边缘的距离
                 R=250,  # 圆半径
                 border=13,  # 字到圆圈内侧的距离
                 r=90,  # 五星外接圆半径
                 fill=(255, 0, 0, 120),  # 印章颜色， 默认纯红色， 透明度0-255，建议90-180
                 ch_font="fonts/SIMSUN.ttf",  # 字体路径， 默认使用windows自带字体
                 en_font="fonts/SIMSUN.ttf",  # 字体路径， 默认使用windows自带字体
                 words_up="测试专用章有限公司",  # 上部文字
                 angle_up=270,  # 上部文字弧形角度
                 font_size_up=80,  # 上部文字大小
                 font_xratio_up=0.66,  # 上部文字横向变形比例
                 stroke_width_up=2,  # 上部文字粗细，一般取值0,1,2,3

                 words_mid="测试专用章",  # 中部文字
                 angle_mid=72,  # 中部文字弧形角度
                 font_size_mid=60,  # 中部文字大小
                 font_xratio_mid=0.7,  # 中部文字横向变形比例
                 stroke_width_mid=1,  # 中部文字粗细，一般取值0,1,2

                 words_down="0123456789",  # 下部文字
                 angle_down=60,  # 下部文字弧形角度
                 font_size_down=20,  # 下部文字大小
                 font_xratio_down=1,  # 下部文字横向变形比例
                 stroke_width_down=1,  # 下部文字粗细，一般取值0,1,2

                 img_wl_path="wl_images/wl0.webp",  # 纹理图路径
                 img_bg_path='bg_images/bg0.jpg',
                 save_path="make_data",  # 保存图片路径
                 save_name='stamp'
                 ):

        self.img_wl_path = img_wl_path # # 纹理图路径
        self.img_bg_path = img_bg_path # 背景图路径
        self.save_path = save_path # 保存图片路径
        self.save_name = save_name

        self.fill = fill  # 印章颜色
        self.ch_font = ch_font  # 字体路径
        self.en_font = en_font  # 字体路径
        self.edge = edge  # 图片边缘空白的距离
        self.H = H  # 圆心到中层文字下边缘的距离
        self.R = R  # 圆半径
        self.r = r  # 五星外接圆半径
        self.border = border  # 字到圆圈内侧的距离

        self.words_up = words_up  # 上部文字
        self.angle_up = angle_up  # 上部文字弧形角度
        self.font_size_up = font_size_up  # 上部文字大小
        self.font_xratio_up = font_xratio_up  # 上部文字横向变形比例
        self.stroke_width_up = stroke_width_up  # 上部文字粗细，一般取值0,1,2,3

        self.words_mid = words_mid  # 中部文字
        self.angle_mid = angle_mid  # 中部文字弧形角度
        self.font_size_mid = font_size_mid  # 中部文字大小
        self.font_xratio_mid = font_xratio_mid  # 中部文字横向变形比例
        self.stroke_width_mid = stroke_width_mid  # 中部文字粗细，一般取值0,1,2,3

        self.words_down = words_down  # 下部文字
        self.angle_down = angle_down  # 下部文字弧形角度
        self.font_size_down = font_size_down  # 下部文字大小
        self.font_xratio_down = font_xratio_down  # 下部文字横向变形比例
        self.stroke_width_down = stroke_width_down  # 中部文字粗细，一般取值0,1,2,3

    def draw_rotated_text(self, image, angle, xy, r, word, fill, font_type, font_size, font_xratio, stroke_width, font_flip=False,
                          *args, **kwargs):
        """
            image:底层图片
            angle：旋转角度
            xy：旋转中心
            r:旋转半径
            text：绘制的文字
            fill：文字颜色
            font_size：字体大小
            font_xratio：x方向缩放比例（印章字体宽度较标准宋体偏窄）
            stroke_width： 文字笔画粗细
            font_flip:文字是否垂直翻转（印章下部文字与上部是相反的）
        """

        # 加载字体文件-直接使用windows自带字体

        font = ImageFont.truetype(font_type, font_size, encoding="utf-8")

        # 获取底层图片的size
        width, height = image.size
        max_dim = max(width, height)
        # 创建透明背景的文字层，大小4倍的底层图片
        mask_size = (max_dim * 2, max_dim * 2)
        # 印章通常使用较窄的字体，这里将绘制文字的图层x方向压缩到font_xratio的比例
        mask_resize = (int(max_dim * 2 * font_xratio), max_dim * 2)
        mask = Image.new('L', mask_size, 0)

        # 在上面文字层的中心处写字，字的左上角与中心对其
        draw = ImageDraw.Draw(mask)

        # 获取当前设置字体的宽高
        bd = draw.textbbox((max_dim, max_dim), word, font=font, align="center", *args, **kwargs)
        font_width = bd[2] - bd[0]
        font_hight = bd[3] - bd[1]
        font_hight = font.size

        # 文字在圆圈上下的方向，需要通过文字所在文字图层的位置修正，保证所看到的文字不会上下颠倒
        if font_flip:
            word_pos = (int(max_dim - font_width / 2), max_dim + r - font_hight)
        else:
            word_pos = (int(max_dim - font_width / 2), max_dim - r)

        # 写字， 以xy为中心，r为半径，文字上边中点为圆周点，绘制文字
        draw.text(word_pos, word, 255, font=font, align="center", stroke_width=stroke_width, *args, **kwargs)
        # 调整角度,对于Π*n/2的角度，直接rotate即可，对于非Π*n/2的角度，需要先放大图片以减少旋转带来的锯齿
        if angle % 90 == 0:
            rotated_mask = mask.resize(mask_resize).rotate(angle)
        else:
            bigger_mask = mask.resize((int(max_dim * 8 * font_xratio), max_dim * 8), resample=Image.BICUBIC)
            rotated_mask = bigger_mask.rotate(angle).resize(mask_resize, resample=Image.LANCZOS)

        # 切割文字的图片
        mask_xy = (max_dim * font_xratio - xy[0], max_dim - xy[1])
        b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
        mask = rotated_mask.crop(b_box)

        # 粘贴到目标图片上
        color_image = Image.new('RGBA', image.size, fill)
        image.paste(color_image, mask)

    def draw_stamp(self):
        # 打开纹理图像，随机截取旋转
        self.img_wl = Image.open(self.img_wl_path)
        # 图像初始设置为None
        self.img = None
        self.join_img = None
        # 创建一张底图,用来绘制文字
        img = Image.new("RGBA", (2 * (self.R + self.edge), 2 * (self.R + self.edge)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # 绘制圆弧， R为外边缘，width往圆心算
        draw.arc(circle(self.R + self.edge, self.R + self.edge, self.R), start=0, end=360, fill=self.fill,
                 width=self.border)

        # 绘制多边形
        if random.uniform(0, 1) < 0.5:
            draw.polygon(pentagram(self.R + self.edge, self.R + self.edge, self.r), fill=self.fill, outline=self.fill)

        # 绘制上圈文字
        angle_word = self.angle_up / len(self.words_up)
        angle_word_curr = ((len(self.words_up) - 1) / 2) * angle_word

        for word in self.words_up:
            self.draw_rotated_text(img, angle_word_curr, (self.R + self.edge, self.R + self.edge),
                                   self.R - self.border * 2,
                                   word, self.fill, self.ch_font, self.font_size_up, self.font_xratio_up, self.stroke_width_up)
            angle_word_curr = angle_word_curr - angle_word

        # 绘制中层文字
        angle_word = self.angle_mid / len(self.words_mid)
        angle_word_curr = -((len(self.words_mid) - 1) / 2) * angle_word

        for word in self.words_mid:
            self.draw_rotated_text(img, 0,
                                   (self.R + self.edge + self.H * tan(angle_word_curr * pi / 180), self.R + self.edge),
                                   self.H,
                                   word, self.fill, self.ch_font, self.font_size_mid, self.font_xratio_mid, self.stroke_width_mid,
                                   font_flip=True)
            angle_word_curr = angle_word_curr + angle_word

        # 绘制下圈文字
        angle_word = self.angle_down / len(self.words_down)
        angle_word_curr = -((len(self.words_down) - 1) / 2) * angle_word

        for word in self.words_down:
            self.draw_rotated_text(img, angle_word_curr, (self.R + self.edge, self.R + self.edge),
                                   self.R - self.border * 2,
                                   word, self.fill, self.en_font, self.font_size_down, self.font_xratio_down, self.stroke_width_down,
                                   font_flip=True)
            angle_word_curr = angle_word_curr + angle_word

        # 随机圈一部分纹理图
        # pos_random = (randint(0, 200), randint(0, 100))
        # box = (pos_random[0], pos_random[1], pos_random[0] + 300, pos_random[1] + 300)
        # img_wl_random = self.img_wl.crop(box).rotate(randint(0, 360))
        # # 重新设置im2的大小，并进行一次高斯模糊
        # img_wl_random = img_wl_random.resize(img.size).convert('L').filter(ImageFilter.GaussianBlur(2))
        # # 将纹理图的灰度映射到原图的透明度，由于纹理图片自带灰度，映射后会有透明效果，所以fill的透明度不能太低
        # L, H = img.size
        # for h in range(H):
        #     for l in range(L):
        #         dot = (l, h)
        #         img.putpixel(dot, img.getpixel(dot)[:3] + (int(img_wl_random.getpixel(dot) / 255 * img.getpixel(dot)[3]*2.5),))
        # # 进行一次高斯模糊，提高真实度

        # 随机旋转
        if random.uniform(0, 1) < 0.5:
            img = img.rotate(randint(0, 360), fillcolor=(255, 255, 255, 0))

        self.img = img.filter(ImageFilter.GaussianBlur(0.6))

    def show_stamp(self):
        if self.img:
            self.img.show()

    def join_stamp(self):
        if self.img:
            background = Image.open(self.img_bg_path)
            foreground = self.img

            b_w, b_h = background.size
            f_w, f_h = foreground.size

            # 进行随机缩放
            scale_factor = random.uniform(0.4, 1)
            foreground = foreground.resize((int(f_w * scale_factor), int(f_h * scale_factor)))
            f_w, f_h = foreground.size

            if f_w > b_w or f_h > b_h:
                background = background.resize((f_w, f_h))
                b_w, b_h = background.size

            x = random.randint(0, b_w - f_w)
            y = random.randint(0, b_h - f_h)

            background.paste(im=foreground, box=(x, y), mask=foreground)
            background = background.crop((x, y, x + f_w, y + f_h))
            self.join_img = background

    def save_stamp(self):
        if self.img:
            # opencv_image = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
            # self.img.save(os.path.join(self.save_path, self.save_name+'.png'))
            self.img.convert("RGB").save(os.path.join(self.save_path, self.save_name+'.jpg'))
            # cv2.imwrite((os.path.join(self.save_path, self.save_name+'.jpg')), opencv_image)

    def save_join_stamp(self):
        if self.join_img:
            # opencv_image = cv2.cvtColor(np.array(self.join_img), cv2.COLOR_RGB2BGR)
            # self.img.save(os.path.join(self.save_path, self.save_name+'_join.png'))
            self.join_img.convert("RGB").save(os.path.join(self.save_path, self.save_name+'_join.jpg'))
            # cv2.imwrite((os.path.join(self.save_path, self.save_name+'_join.jpg')), opencv_image)


class Words:
    def __init__(self, dict_path) -> None:
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_list = [i.replace('\n', '') for i in f.readlines()]

    def get_txt_chinese(self, char_len = (4, 8)):
        weight = [1]*len(self.char_list)
        txt = choices(self.char_list, weight, k=randint(5, 6))
        txt = ''.join(txt)
        if len(txt) == 1:
            len_txt = 1
        else:
            len_txt = randint(char_len[0], char_len[1])
        start_txt = randint(0,len(txt)-len_txt)
        end_txt = start_txt+len_txt
        txt = txt[start_txt:end_txt]
        txt = txt[:20]
        return txt

    def get_txt_number(self, char_len = (3, 10)):
        char_list = ['0','1','2','3','4','5','6','7','8','9']
        weight = [1]*len(char_list)
        txt = choices(char_list, weight, k=randint(char_len[0],char_len[1]))
        txt = ''.join(txt)
        return txt

def main(pid, queue, kwargs):
    make_path = kwargs['make_path']
    up_num = kwargs['up_num']
    mid_num = kwargs['mid_num']
    down_num = kwargs['down_num']
    dict_path = kwargs['dict_path']
    wl_image_path_list = kwargs['wl_image_path_list']
    bg_image_path_list = kwargs['bg_image_path_list']
    font_path_list = kwargs['font_path_list']
    font_xratio_up_dict = kwargs['font_xratio_up_dict']
    font_xratio_mid_dict = kwargs['font_xratio_mid_dict']

    words_process = Words(dict_path)
    stamp = Stamp()

    stamp.save_path = os.path.join(make_path, str(pid))
    os.makedirs(stamp.save_path, exist_ok=True)

    print(f'进程:{pid} 初始化完成')

    while(True):
        if queue.empty():
            print(f'进程:{pid} 队列为空')
            break
        i = queue.get()
        print(f'进程:{pid} 开始生成印章 {i} 剩余印章 {queue.qsize()}')
        try:
            words_up = words_process.get_txt_chinese(char_len=up_num)
            words_mid = words_process.get_txt_chinese(char_len=mid_num)
            words_down = words_process.get_txt_number(char_len=down_num)

            with open(os.path.join(make_path, f'save_{pid}.txt'), 'a', encoding='utf-8') as f:
                f.write(f'{pid}/{i}.jpg\t{words_up}#{words_mid}#{words_down}\n')
                f.write(f'{pid}/{i}_join.jpg\t{words_up}#{words_mid}#{words_down}\n')

            # stamp.words_up = words[0]
            # stamp.words_mid = words[1]
            # stamp.words_down = words[2]

            #############设置随机参数#############
            # stamp.img_wl_path = img_wl_path # 纹理图路径!!!!!!!!!!!!

            # stamp.save_name = save_name
            red_color = (255-random.randint(0,10), 0+random.randint(0,10), 0+random.randint(0,10), 180+random.randint(-10,10))
            # gray_color_thres = random.randint(-50,100)
            # gray_color = (128+gray_color_thres, 128+gray_color_thres, 128+gray_color_thres,120+random.randint(-10,10))


            stamp.fill = red_color  # 印章颜色(红色或灰色)
            stamp.edge = random.randint(4, 6)  # 图片边缘空白的距离 5
            # stamp.H = random.randint(157,163)  # 圆心到中层文字下边缘的距离 default 160
            stamp.R = random.randint(245,255)  # 圆半径 default 250
            stamp.r = random.randint(85, 95)  # 五星外接圆半径 default 90
            stamp.border = random.randint(12, 14)  # 字到圆圈内侧的距离 default 13

            stamp.words_up = words_up  # 上部文字
            stamp.angle_up = random.randint(265,275)  # 上部文字弧形角度 default 270
            stamp.font_size_up = random.randint(77,83)  # 上部文字大小 default 80
            stamp.font_xratio_up = font_xratio_up_dict[str(len(words_up))]  # 上部文字横向变形比例
            stamp.stroke_width_up = random.randint(1,3)  # 上部文字粗细，一般取值0,1,2,3 默认2

            stamp.words_mid = words_mid  # 中部文字
            stamp.angle_mid = random.randint(67,77)  # 中部文字弧形角度 default 72
            stamp.font_size_mid = random.randint(57,63)  # 中部文字大小 default 60
            stamp.font_xratio_mid = font_xratio_mid_dict[str(len(words_mid))]  # 中部文字横向变形比例
            # stamp.stroke_width_mid = random.randint(1,2)  # 中部文字粗细，一般取值0,1,2,3 默认1

            stamp.words_down = words_down  # 下部文字
            stamp.angle_down = random.randint(59,61)  # 下部文字弧形角度 default 60
            stamp.font_size_down = random.randint(18,22)  # 下部文字大小 default 20
            # stamp.font_xratio_down = font_xratio_down  # 下部文字横向变形比例
            # stamp.stroke_width_down = random.randint(1,2)  # 下部文字粗细，一般取值0,1,2,3 默认1
            ##############
            stamp.img_wl_path = random.choice(wl_image_path_list)
            stamp.ch_font = random.choice(font_path_list)

            stamp.img_bg_path = random.choice(bg_image_path_list)
            stamp.save_name = str(i)
            
            stamp.draw_stamp()
            stamp.join_stamp()
            stamp.save_stamp()
            stamp.save_join_stamp()
            print(f'进程:{pid} 保存印章成功 {i}')
        except Exception as e:
            print(f'进程:{pid} 保存印章失败 {i} {e}')

    print(f'进程:{pid} 结束')


if __name__ == "__main__":
    process_num = 4 # 设置进程数
    seal_num = 40 # 生成印章数量
    up_num = (4,20) # 上圈文字个数范围
    mid_num = (3,5) # 中间行文字个数范围
    down_num = (3,8) # 下圈数字个数范围

    dict_path = 'dicts/hospital.txt' # 字典路径
    make_path = 'make_data' # 生成数据保存路径
    wl_path = "wl_images" # 纹理图片路径（没有使用）
    bg_path = "bg_images" # 背景图片路径
    font_path = "fonts" # 字体文件路径
    txt_name = 'make_data.txt' # 生成数据保存文件名

    wl_image_path_list = get_files_list(wl_path)
    bg_image_path_list = get_files_list(bg_path)
    font_path_list = get_files_list(font_path)

    font_xratio_up_dict = {'3': 0.7, '4': 0.685, '5': 0.67, '6': 0.655, '7': 0.64, '8': 0.625, '9': 0.61, '10': 0.595, '11': 0.58, '12': 0.565, '13': 0.55, '14': 0.535, '15': 0.5, '16': 0.5, '17': 0.5, '18': 0.5, '19': 0.5, '20': 0.5}
    font_xratio_mid_dict = {'3': 0.7, '4': 0.65, '5': 0.6, '6': 0.54, '7': 0.5, '8': 0.5, '9': 0.5}

    queue = Queue(maxsize=0)

    for index in range(seal_num):
        queue.put(index)

    input_kwargs = {
        'make_path': make_path,
        'up_num' : up_num,
        'mid_num' : mid_num,
        'down_num' : down_num,
        'dict_path': dict_path,
        'wl_image_path_list': wl_image_path_list, 
        'bg_image_path_list': bg_image_path_list, 
        'font_path_list': font_path_list, 
        'font_xratio_up_dict': font_xratio_up_dict, 
        'font_xratio_mid_dict': font_xratio_mid_dict
    }

    process_list = []
    for pid in range(process_num):
        p = Process(target=main, args=(pid, queue, input_kwargs), daemon=True)
        p.start()
        process_list.append(p)
    
    for p in process_list:
        p.join()

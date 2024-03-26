import os
import cv2
import numpy as np
from math import pi, cos, sin, tan
from random import randint, choices
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from multiprocessing import Process, Queue


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

                 img_wl_path="background.webp",  # 纹理图路径
                 save_path="make_data",  # 保存图片路径
                 save_name='stamp'
                 ):

        self.img_wl_path = img_wl_path # # 纹理图路径
        self.save_path = save_path # 保存图片路径
        self.save_name = save_name

        self.fill = fill  # 印章颜色
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

    def draw_rotated_text(self, image, angle, xy, r, word, fill, font_size, font_xratio, stroke_width, font_flip=False,
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

        # 加载字体文件-直接使用windows自带字体，中文用simsun， 英文用arial
        if is_Chinese(word):
            font = ImageFont.truetype("SIMSUN.ttf", font_size, encoding="utf-8")
        else:
            font = ImageFont.truetype("arialr.ttf", font_size, encoding="utf-8")

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
            bigger_mask = mask.resize((int(max_dim * 8 * font_xratio), max_dim * 8),
                                      resample=Image.BICUBIC)
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
        # 创建一张底图,用来绘制文字
        img = Image.new("RGBA", (2 * (self.R + self.edge), 2 * (self.R + self.edge)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # 绘制圆弧， R为外边缘，width往圆心算
        draw.arc(circle(self.R + self.edge, self.R + self.edge, self.R), start=0, end=360, fill=self.fill,
                 width=self.border)

        # 绘制多边形
        draw.polygon(pentagram(self.R + self.edge, self.R + self.edge, self.r), fill=self.fill, outline=self.fill)

        # 绘制上圈文字
        angle_word = self.angle_up / len(self.words_up)
        angle_word_curr = ((len(self.words_up) - 1) / 2) * angle_word

        for word in self.words_up:
            self.draw_rotated_text(img, angle_word_curr, (self.R + self.edge, self.R + self.edge),
                                   self.R - self.border * 2,
                                   word, self.fill, self.font_size_up, self.font_xratio_up, self.stroke_width_up)
            angle_word_curr = angle_word_curr - angle_word

        # 绘制中层文字
        angle_word = self.angle_mid / len(self.words_mid)
        angle_word_curr = -((len(self.words_mid) - 1) / 2) * angle_word

        for word in self.words_mid:
            self.draw_rotated_text(img, 0,
                                   (self.R + self.edge + self.H * tan(angle_word_curr * pi / 180), self.R + self.edge),
                                   self.H,
                                   word, self.fill, self.font_size_mid, self.font_xratio_mid, self.stroke_width_mid,
                                   font_flip=True)
            angle_word_curr = angle_word_curr + angle_word

        # 绘制下圈文字
        angle_word = self.angle_down / len(self.words_down)
        angle_word_curr = -((len(self.words_down) - 1) / 2) * angle_word

        for word in self.words_down:
            self.draw_rotated_text(img, angle_word_curr, (self.R + self.edge, self.R + self.edge),
                                   self.R - self.border * 2,
                                   word, self.fill, self.font_size_down, self.font_xratio_down, self.stroke_width_down,
                                   font_flip=True)
            angle_word_curr = angle_word_curr + angle_word

        # 随机圈一部分纹理图
        pos_random = (randint(0, 200), randint(0, 100))
        box = (pos_random[0], pos_random[1], pos_random[0] + 300, pos_random[1] + 300)
        img_wl_random = self.img_wl.crop(box).rotate(randint(0, 360))
        # 重新设置im2的大小，并进行一次高斯模糊
        img_wl_random = img_wl_random.resize(img.size).convert('L').filter(ImageFilter.GaussianBlur(1))
        # 将纹理图的灰度映射到原图的透明度，由于纹理图片自带灰度，映射后会有透明效果，所以fill的透明度不能太低
        L, H = img.size
        for h in range(H):
            for l in range(L):
                dot = (l, h)
                img.putpixel(dot,
                             img.getpixel(dot)[:3] + (int(img_wl_random.getpixel(dot) / 255 * img.getpixel(dot)[3]),))
        # 进行一次高斯模糊，提高真实度
        self.img = img.filter(ImageFilter.GaussianBlur(0.6))

    def show_stamp(self):
        if self.img:
            self.img.show()

    def save_stamp(self):
        if self.img:
            opencv_image = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
            self.img.save(os.path.join(self.save_path, self.save_name+'.png'))
            cv2.imwrite((os.path.join(self.save_path, self.save_name+'.jpg')), opencv_image)

class Words:
    def __init__(self, ch_dict_path) -> None:
        with open(ch_dict_path, 'r', encoding='utf-8') as f:
            self.char_list = [i.replace('\n', '') for i in f.readlines()]

    def get_txt_chinese(self, char_len = (4, 8)):
        weight = [1]*len(self.char_list)
        txt = choices(self.char_list, weight, k=randint(3, 4))
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


def main(pid, queue, make_path):
    stamp = Stamp()
    stamp.save_path = make_path
    print(f'进程:{pid} 初始化完成')
    while(True):
        if queue.empty():
            print(f'进程:{pid} 结束 队列为空')
            break
        i, words = queue.get()
        print(f'进程:{pid} 开始生成印章 {i}')
        stamp.words_up = words[0]
        stamp.words_mid = words[1]
        stamp.words_down = words[2]
        stamp.img_wl_path = "background.webp"
        stamp.save_name = str(i)
        stamp.draw_stamp()
        # stamp.show_stamp()
        stamp.save_stamp()
        print(f'进程:{pid} 保存印章成功 {i}')


if __name__ == "__main__":
    # 设置进程数
    process_num = 4
    process_list = []
    
    ch_dict_path = 'all_dicts/hospital.txt'
    make_path = 'make_data'

    words_process = Words(ch_dict_path)
    queue = Queue(maxsize=0)

    os.makedirs(make_path, exist_ok=True)
    f = open(os.path.join(make_path, 'all.txt'), 'w', encoding='utf-8')
    for index in range(30):
        words_up = words_process.get_txt_chinese(char_len=(4, 8))
        words_mid = words_process.get_txt_chinese(char_len=(2,5))
        words_down = words_process.get_txt_number(char_len=(3,10))
        words = (words_up, words_mid, words_down)
        queue.put((index, words))
        f.write(f'{index}.png\t{words_up}#{words_mid}#{words_down}\n')
        f.write(f'{index}.jpg\t{words_up}#{words_mid}#{words_down}\n')
    f.close()

    for pid in range(process_num):
        p = Process(target=main, args=(pid, queue, make_path), daemon=True)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
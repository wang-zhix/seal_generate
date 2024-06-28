## 印章生成

1、参考dicts/hospital.txt添加你的字典到dicts文件夹中

2、bg_images可以添加背景图

3、fonts可以添加字体文件

4、修改seal_generate.py的__main__函数
```
process_num = 4 # 设置进程数
seal_num = 40 # 生成印章数量
up_num = (4,20) # 上圈文字个数范围
mid_num = (3,9) # 中间行文字个数范围
down_num = (3,8) # 下圈数字个数范围

dict_path = 'dicts/hospital.txt' # 字典路径
make_path = 'make_data' # 生成数据保存路径
wl_path = "wl_images" # 纹理图片路径（没有使用）
bg_path = "bg_images" # 背景图片路径
font_path = "fonts" # 字体文件路径
```
5、运行seal_generate.py
```
python seal_generate.py
```


## 增加内容

1、从字典中随机获取文字

2、启动多进程生成

3、生成对应标签文件all.txt

4、纹理图暂时没有使用

## 项目引用

地址 https://github.com/zh3389/seal_generate.git

使用Python的PIL库绘图

1、先画个圆；

2、画个五角星；

3、绘制文字；

4、增加纹理效果（需要下载附件纹理图并修改代码中纹理图路径），看上去更真实。

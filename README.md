# RJB
第八届中国软件杯三等奖——基于深度学习的银行卡号识别
<br>项目采用两阶段法进行识别，目标检测部分复现了U-Net，卡号识别部分复现了CRNN。</br>

<h3>主程序启动方式</h3>

<br>运行:python3 demo.py</br>
<br>注意:需识别的图片集放在当前目录的img_dir文件夹下，识别结果保存至./crop/result.txt</br>
<br>目标检测的过程可视化保存在crop文件夹中，可以看到经过检测裁剪出来的银行卡区域。</br>

<h3>数据标注代码说明</h3>

![Image text](https://github.com/HuiyanWen/RJB/blob/master/annotation.png)

<h3>目标检测代码说明</h3>

![Image text](https://github.com/HuiyanWen/RJB/blob/master/iam.png)

<h3>CRNN代码说明</h3>

![Image text](https://github.com/HuiyanWen/RJB/blob/master/crnn.png)


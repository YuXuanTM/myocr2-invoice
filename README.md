# myocr2-invoice

#### 介绍
发票OCR识别，实现方式使用YOLOv10提取关键位置发票信息，PaddleOCR根据提取的位置进行文字识别。
支持图片和PDF识别，主要识别了发票标题、发票代码、发票号码、开票日期、购买方名称、购买方识别号、销售方名称、销售方识别号、含税金额、不含税金额、税费信息。

#### 软件架构
YOLOv10+PaddleOCR+Flask


#### 安装教程

1. pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
2. 安装PaddleOCR所需内容，参考https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html
例如：
 2.1 cpu环境
  python -m pip install paddlepaddle==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
 2.1 gpu环境以英伟达显卡CUDA11.7为例 
  python -m pip install paddlepaddle-gpu==2.6.1.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
3. pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
4. python main.py
启动成功发送地址测试http://127.0.0.1:5000/invoice_ocr

#### 测试截图
经测试GPU环境下平均三百多毫秒，CPU环境下一秒左右
![输入图片说明](https://foruda.gitee.com/images/1733117962971085706/7da738f8_5748498.png "GPU环境平均耗时")
![输入图片说明](https://foruda.gitee.com/images/1733118115493827803/c57188ac_5748498.png "CPU环境平均耗时")

![输入图片说明](https://foruda.gitee.com/images/1733117764446225949/6cdf117d_5748498.png "屏幕截图")

#### 后续
目前训练数据种类比较少，后续逐步完善。


#### 说明

若要商用请注意YOLOv10开源协议以及PaddleOCR开源协议，项目地址如下
YOLOv10：https://github.com/THU-MIG/yolov10
PaddleOCR：https://github.com/PaddlePaddle/PaddleOCR


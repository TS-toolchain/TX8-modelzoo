# 模型信息

Resnet（Residual Network， 残差网络）系列网络是一种在图像分类中广泛使用的深度卷积神经网络，同时也作为计算机视觉任务主干经典神经网络的一部分。典型的resnet网络有resnet50, resnet101等。

论文：[《Deep Residual Learning for Image Recognition》](https://arxiv.org/abs/1512.03385)

开源模型链接：https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/resnet.py

# 资源准备

1) 1台REX1032服务器(卡满插且卡均正常)。
2) 清微docker镜像：TX8100_REX1032_Release_v4.3.7.p3.tar.gz。
3) 模型资源下载(可选)
   软件包中已有resnet50.onnx 模型，数据类型FP16, 可直接使用。
   也可选择下载官网模型。[下载resnet50 onnx权重](https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx?download=)

4) 工具链TxNN软件包：txnn_1.2.0_buildxxx.tar.gz，解压后，与该示例相关的内容如下：

   其中：
    * yolov5s文件夹中包含运行yolov5s模型相关的模型，数据和脚本。
    * script文件夹中 deploy文件夹为环境部署文件夹，其下install_vllm.sh文件为部署CV模型执行文件；
    * txnn.1.2.0_build2025xxx.tar.gz压缩包为推理引擎版本包。

# 准备环境

## 启动device

用户在Host端部署执行环境

1) 使用终端工具ssh登录REX1032服务器

   执行如下命令行：

   ```shell
   # HOST_IP为REX1032服务器IP，用户需替换为实际IP地址数值
   ssh -X user@HOST_IP
   ```

2) 解压清微docker镜像压缩包

   执行如下命令行解压：

   ```shell
   sudo tar -zxf TX8100_REX1032_Release_v4.3.7.p3.tar.gz
   ```

   解压得到文件：
   `TX8100_REX1032_Release_v4.3.7.p3.tar`
   `Tsm_driver_4.3.7.P3_x86_64_installer.run`


3) 执行启动Device

   拉起TX8 Device并等待ready，新开会话窗口2，执行如下命令：
   ```shell
   ./Tsm_driver_4.3.7.P3_x86_64_installer.run install silicon
   ```

## 环境部署

1) 解压工具链TxNN软件包
   将txnn_1.2.0_buildxxx.tar.gz放至登录用户指定目录下，如用户根目录(~/，下文中绝对路径/login_home)，并解压：

```shell
cd ~/
tar -xzvf txnn_1.2.0_buildxxx.tar.gz
```

解压后得到txnn_1.2.0_buildxxx目录。

2) 加载docker镜像
   执行如下命令行：

```shell
sudo docker load -i TX8100_REX1032_Release_v4.3.7.p3.tar
```

Load完毕可以使用如下命令查看镜像是否load成功：

```shell
sudo docker images
```

3) 创建docker容器
   执行如下命令创建容器：

```shell
sudo docker run -d --name txnn --network=host --ipc=host -v /dev:/dev -v /tmp:/tmp
-v /lib/modules:/lib/modules -v /sys:/sys -v /login_home:/login_home hub.tsingmicro.com/tx8/v4.3.7.p3:
kuiper-rex1032-release  
```

4) 进入docker容器
   执行如下命令进入容器：

```shell
sudo docker exec -it txnn /bin/bash
```

5) 执行环境部署脚本
6)

进入登录用户home目录，带参数执行install_vllm.sh脚本，共需要传入5个参数，分别为：1. HOME_PATH 目录；2. VERSION版本号；3.
MODEL_PATH模型所在路径；4.模型数据类型；5.chip_out所在路径。
示例如下

```shell
cd $HOME_PATH/txnn_1.2.0_buildxxx/script/deploy
#带参数运行脚本

bash install_vllm.sh /login_home/xxx txnn.1.2.0buildxxx /login_home/xxx/DeepSeek-R1-Distill-Qwen-7B bf16
/login_home/xxx/deepseek_7b_bf16_seq8192_c4/chip_out/node_0_0/
```

6) 编译onnx模型生成板端运行文件

   解压`script/TxCompiler.1.2.0_buildxxx.tar.gz` 工具，直接执行其中的sh命令接口，示例如下：

```shell
  tar -xzvf TxCompiler.1.2.0_buildxxx.tar.gz
  cd TxCompiler
  #将onnx模型作为入参
  bash scripts/TxCompiler.sh ../../resnet50/model/resnet50_fp16.onnx
 ```

## 图像分类推理

`yolov5s/plot_image`目录下提供有脚本：plot_image.py，供用户在docker内执行，体验推理功能。
运行示例：在`REX1032`服务器上，推理图片，并绘制检测框。
在会话窗口1执行如下命令，指定`--device`参数选择卡运行，默认0号卡。

```shell
cd $HOME_PATH/txnn_1.2.0_buildxxx/yolov5s/plot_image
python plot_image.py
```

执行成功后可输出画框结果图保存在`./runs/detect`目录下。

## mAP精度评估

yolov5s/eval_coco目录下提供有脚本：eval_yolov5s_coco.py
运行示例：在REX1032服务器，评估coco数据集精度。
在会话窗口1执行如下命令，启动批量数据推理服务，指定--device参数选择卡运行，默认0号卡。

```shell
cd $HOME_PATH/txnn_1.2.0_buildxxx/yolov5s/eval_coco
python eval_yolov5s_coco.py
```

执行成功，输出mAP结果和推理性能结果，界面如下所示：
![img_yolov5s](img_yolov5s.png)

# 版本说明

2025/3/7 第一版

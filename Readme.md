### tools：
实现参考自： https://github.com/pureyangcry/tools

- `Attack：`
  - `Attack.py` 用于给测试文件添加攻击，包含高斯噪声，椒盐噪声，粉红噪声，JPEG压缩，均值模糊，高斯模糊，直方图均衡化

  - `colorednoise.py` 用于生成粉红噪声

- `DataAugForObjectSegmentation：`
  - `DataAugForObjectSegmentation.py` 用于训练数据增广，包含亮度改变，平移，翻转


### PaddleSeg：
- `clone` 自 https://github.com/PaddlePaddle/PaddleSeg  的 `release/2.6` 分支

### 数据格式：
  ```
  coco
    |
    |--images
    |  |--train
    |  |--val
    |
    |--annotations
    |  |--train
    |  |--val
  ```

### 修改的文件有：
- `PaddleSeg/paddleseg/datasets/cocostuff.py`
  - `75、77` 行的图片格式（`.jpg -> .png`

- `PaddleSeg/configs/_base_/coco_stuff.yml `
- `PaddleSeg/configs/pointrend/pointrend_resnet50_os8_cityscapes_1024×512_80k.yml  `
  - 设置训练相关参数 （数据来自文件夹 `PaddleSeg/data/cocostuff`

- `PaddleSeg/paddleseg/models/backbones/resnet_vd.py`
  - 添加了 `SENet` 模块， `246` 行设为 `True` 为加入 `SENet` （有一个 `resnet_vd-old.py` 为原始代码

- `segInPaper/PaddleSeg/paddleseg/transforms/transforms.py`
  - `856` 行添加了各种攻击的代码，训练需要加入攻击时在 `PaddleSeg/configs/_base_/coco_stuff.yml` 文件的 `- type: Normalize` 同级处添加 `- type: RandomAttack`


### 添加的文件
- `PaddleSeg/configs/_base_/coco_stuff_copy.yml `
- `PaddleSeg/configs/pointrend/pointrend_resnet50_os8_cityscapes_1024×512_80k_copy.yml`
  - `predict.py` 和 `val.py` 相关参数 （数据文件来自文件夹 `PaddleSeg/data/cocoval`


### 训练好的模型
- 测试均使用了 `output-200-xxxxx/best_model` 文件夹内的模型

- `output-200-6       ` - 原始       
- `output-200-6-se    ` - 仅加注意力 
- `output-200-6-atk   ` - 仅加攻击层 
- `output-200-6-se-atk` - 都加       


### 使用的命令示例
- 相关参数的意义可参考：https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/whole_process_cn.md

  ```
  CUDA_VISIBLE_DEVICES=1 python train.py \
    --config configs/pointrend/pointrend_resnet50_os8_cityscapes_1024×512_80k.yml \
    --do_eval \
    --use_vdl \
    --batch_size 6 \
    --iters 200000 \
    --save_interval 2000 \
    --save_dir ./output-200-6-se-atk \
    --keep_checkpoint_max 20 


  CUDA_VISIBLE_DEVICES=0 python val.py \
  --config configs/pointrend/pointrend_resnet50_os8_cityscapes_1024×512_80k_copy.yml \
  --model_path output-200-6-se-atk/best_model/model.pdparams


  CUDA_VISIBLE_DEVICES=0 python predict.py \
    --config configs/pointrend/pointrend_resnet50_os8_cityscapes_1024×512_80k_copy.yml\
    --model_path output-200-6-se-atk/best_model/model.pdparams \
    --image_path data/cocoval/images/test \
    --save_dir output-200-6-8/test

  ```
### 其他
- 使用 `labelme` 标注  `->`  使用数据增广  `->`  将标签转换为训练格式 （`PaddleSeg/tools/labelme2seg.py` 和 `convert_cocostuff.py`

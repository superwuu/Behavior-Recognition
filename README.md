# 【算法挑战赛】基于无人机的人体行为识别

- 百度网盘链接：https://pan.baidu.com/s/1d2Qv5nMdV1-B2IFo3nXHhg?pwd=inuc 

  ------

  
- **仓库结构：**
  - model：比赛使用的代码仓库
    - ICMEW2024-Track10
      - Model_infrence目录下为Mix_Former和Mix_GCN的代码
  - emsemble：比赛使用模型集成代码仓库
    - run2getW.sh：获取20个模型权重的脚本，启动get_weight.py程序
    - run2getRes.sh：推理testB的脚本，得到pred.npy，启动ensemble_eval.py
    - final_merged_data_select.csv：20个模型对模糊集的分类结果
    - vague_set_optimize.py：模糊集的优化筛选过程
  - 参数文档.pdf：说明文档

    ------
  
    
  
- **百度网盘内容结构说明：**
  - data：训练数据和testB数据的npz文件，其中训练数据测试集为真实标签，testB数据为人工生成的相同标签
  - 1028res：20个模型对testB的预测结果
  - 1028csv：对1028res处理后的csv文件，每个模型对样本的分类结果
  - testA：20个模型在testA上的结果与testA的标签
  - testB：20个模型在testB上的结果（无标签）
  - model_weight：20个模型的最佳权重
  - model_config：20个模型训练与推理时配置文件

    ------
  
    
  
- **运行方式：**
  - 模型训练与推理
    - 训练：
      1. 将训练集整理成npy文件（TE-GCN使用）与npz文件（Top仓库使用）后放置在指定位置
      2. 训练Top仓库：分别运行Mix_Former和Mix_GCN文件夹下的train.sh，结果保存在output文件夹中
      3. 训练TE-GCN仓库：分别运行TRAIN_V1_bone_enhance.sh脚本与TRAIN_V1_joint_enhance.sh脚本，结果保存在work_dir文件夹中
    - 推理：
      1. 将testB同样整理成npy文件与npz文件
      2. 推理Top仓库：别运行Mix_Former和Mix_GCN文件夹下的testB.sh，结果保存在output-B文件夹中
  - 模型集成
    - 第一阶段：
      1. 将testA、testB文件夹放置到ensemble文件夹下
      2. 运行run2getW.sh获取最优权重
      3. 将权重配置到ensemble_eval.py，运行run2getRes.sh脚本得到pred.npy文件
    - 第二阶段：
      1. 修改pkl2npy.ipynb文件，将testB中的pkl文件转成npy文件，得到百度网盘上1028res文件夹下的文件
      2. 运行npy2csv_20model.py，将1028res文件夹下的文件转成1028csv下的文件，为20个模型对testB样本的分类结果
      3. 根据模糊集的概念手动挑选模糊集，得到final_merged_data_select.csv
      4. 运行vague_set_optimize.py，输出最后的pred.npy文件
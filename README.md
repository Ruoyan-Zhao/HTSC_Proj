<!--
 * @Author: WenhuZhang 358607757@qq.com
 * @Date: 2022-12-27 15:48:23
 * @LastEditors: WenhuZhang 358607757@qq.com
 * @LastEditTime: 2022-12-27 16:34:56
 * @FilePath: /project/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 多目标内容推荐

## **1. 环境依赖**
- pandas==1.0.5
- numpy==1.19.5
- numba==0.53.1
- scipy==1.5.0
- torch==1.4.0
- python==3.6.5
- deepctr-torch==0.2.7
- transformers==3.1.0

## **2. 数据准备**
### 2.1 下载原始数据、配置环境
### 2.2 运行prepare相关代码，生成特征文件
```
cd prepare/
python Step1_feed_text_process.py        数据预处理、对齐、补全、TfIDF、特征压缩
python Step2_feed_text_cluster.py        为数据tag、kw生成聚类结果
python Step3_w2v_feed_author_user.py     对tag、kw、feed、user进行word2vec编码生成特征
（以上脚本中的文件路径需手动修改，以适应本地环境）
```
### 2.3 处理完成后数据文件结构如下
```
data
├── raw_data ,下载的原始数据
    ├── user_action.csv
    ├── feed_info.csv
    ├── feed_embeddings.csv
├── my_feed_data ,脚本生成的数据和特征
    ├── label_encoder_models      标签编码工具
            ├──lbe_dic_all.pkl
        ├── official_embed_pca    视频多模态特征pca降维
            ├──......
        ├── feedid_text_features  视频特征预处理结果
            ├──......
        ├── w2v_models            word2vec向量编码模型
            ├──......
```

## **3. 模型训练**
```
cd train/
python generate_train_dat.py    划分训练验证测试集、并预处理模型所用特征
python train.py                 训练模型
（以上脚本中的文件路径需手动修改，以适应本地环境)
```

## **4. 实验结果**
详见exp目录，其中各.log文件为模型迭代记录对应的实验log

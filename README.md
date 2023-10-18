# KGMNER
命名实体识别是各个领域自然语言处理任务的基础，但在特定领域中，中文实体识别由于句子简短或存在歧义等问题变得困难。针对此问题，提出了一种利用学科图谱和图像提高实体识别准确率的模型，旨在利用领域图谱和图像，提高计算机学科领域短文本中实体识别的准确率。使用基于BERT-BiLSTM的模型提取文本特征，使用ResNet152提取图像特征，并使用分词工具获得句子中的名词实体。通过BERT将名词实体与图谱节点进行特征嵌入，使用余弦相似度查找句子中的分词在学科图谱中最相似的节点，保留此节点领域为1的邻居节点，生成最佳匹配子图，作为句子的语义补充。模型使用多层感知机将文本、图像和子图三种特征映射到同一空间，并通过独特的门结构实现文本和图像的细粒度跨模态特征融合。最后通过交叉注意力机制将多模态特征与子图特征进行融合，输入解码器进行实体标记。

#### 目录结构描述
├── Readme.md                   // help
├── app                         // 应用
├── config                      // 配置
│   ├── default.json
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境

#### 部分代码来自：
  - [UMGF](https://github.com/TransformersWsz/UMGF/tree/main)
  - [UMT](https://github.com/jefferyYu/UMT/)
#### Twitter数据：
  - [Twitter](https://github.com/jefferyYu/UMT/)
#### 领域数据：
  - 由于相关研究还在进行，暂不方便公开

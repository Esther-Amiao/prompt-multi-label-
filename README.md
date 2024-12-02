😺Hi,I'm Esther, and this is a project using prompt for text multi-label task in codereview.

这个项目使用prompt范式完成了一个软件社区的多标签多分类问题，并提供了一个多标签多分类任务的解决方案
 1. 首先你需要找到数据集中不存在于预训练语言模型PLM的词
 2. 将这些词加入到模型词表vocab.txt中
 3. 为任务设计一个描述性提示模板，其中需要设置多个MASK掩码位置

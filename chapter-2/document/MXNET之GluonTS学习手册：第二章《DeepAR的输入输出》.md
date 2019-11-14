# MXNET之GluonTS学习手册：第二章《DeepAR的输入/输出》

> 阅读本手册需要一定mxnet、gluon操作基础。 
> 本文使用cpu训练代码。
> 个人博客地址：[https://zmkwjx.github.io](https://zmkwjx.github.io)
> 本文github地址：[https://github.com/zmkwjx/GluonTS-Learning-in-Action](https://github.com/zmkwjx/GluonTS-Learning-in-Action)
> GluonTS官网地址：[https://gluon-ts.mxnet.io](https://gluon-ts.mxnet.io)

## 1、DeepAR
在GluonTS中，**DeepAR实现了一种基于RNN的模型，使用自回归递归网络进行概率预测，是一种在大量相关时间序列上训练自回归递归网络模型的基础上，用于产生准确概率预测的方法**。与最新技术相比，其准确性提高了15％左右。
 概率预测（即根据时间序列的过去来估计时间序列的未来的概率分布）是优化业务流程的关键因素。
* 注意：此模型的代码与[SageMaker的DeepAR预测算法](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)背后的实现无关 

## 2、DeepAR的输入/输出
DeepAR支持两个数据通道。所需的train通道描述了训练数据集。可选test通道描述了算法用于训练后评估模型准确性的数据集。您可以采用JSON行格式提供训练和测试数据集。
指定训练和测试数据的路径时，可以指定一个文件或包含多个文件的目录，这些文件可以存储在子目录中。如果指定目录，则DeepAR会将目录中的所有文件用作相应通道的输入。默认情况下，DeepAR模型使用 **.json文件** 输入数据。

 - **载入数据集的方法**

```python
# 后面将对该方法进行介绍
common.FileDataset("此处填入训练数据文件夹的绝对路径", freq="H")
```

 **2.1 输入数据字段格式**
 - **start** — 格式为 yyy-MM-DD HH:MM:SS 的字符串。开始时间戳不能包含时区信息。
 - **target** — 表示时间序列的浮点值或整数数组。您可以将丢失的值编码为null，或者在JSON中编码为"NAN"字符串：
```python
{"start": "2009-11-01 00:00:00", "target": [5, "NAN", 7, 12]}
```

 - **feat_dynamic_real (可选)** — 代表自定义要素时间序列（动态要素）向量的浮点值或整数数组。如果设置此字段，则所有记录必须具有相同数量的内部数组（相同数量的特征时间序列）。此外，每个内部数组必须具有与关联target值相同的长度 。例如，如果目标时间序列代表不同产品的需求，则feat_dynamic_real可能是布尔时间序列，它指示是否对特定产品应用了促销：
```python
{"start": ..., "target": [5, "NAN", 7, 12], "dynamic_feat": [[1, 0, 0, 1]]}
```

 - **feat_static_cat (可选)** — 可以用于对记录所属的组进行编码的分类特征数组。分类要素必须编码为基于0的正整数序列。例如，分类域{R，G，B}可以编码为{0，1，2}。来自每个分类域的所有值都必须在训练数据集中表示。

 **如果您使用JSON文件，则该文件必须为JSON Lines格式。例如：**
```python
{"start": "2009-11-01 00:00:00", "target": [4.3, "NaN", 5.1, ...], "feat_static_cat": [0, 1], "feat_dynamic_real": [[1.1, 1.2, 0.5, ...]]}
{"start": "2012-01-30 00:00:00", "target": [1.0, -5.0, ...], "feat_static_cat": [2, 3], "feat_dynamic_real": [[1.1, 2.05, ...]]}
{"start": "1999-01-30 00:00:00", "target": [2.0, 1.0], "feat_static_cat": [1, 4], "feat_dynamic_real": [[1.3, 0.4]]}
```
在此示例中，每个时间序列都有两个关联的分类特征和一个时间序列特征。

如果对算法进行无条件训练，它将学习一个“全局”模型，该模型在推理时与目标时间序列的特定身份无关，并且只受其形状的约束。

如果该模型以提供给每个时间序列的 feat_static_cat 和 feat_dynamic_real 特征数据为条件，则预测很可能受到具有相应 cat 特征的时间序列特征的影响。例如，如果 target 时间序列表示服装商品需求，则您可以关联一个二维 cat 向量，该向量在第一个组件中编码商品类型（例如，0 = 鞋子，1 = 连衣裙），在第二个组件中编码商品颜色（例如，0 = 红色，1 = 蓝色）。示例输入如下所示：

```python
{ "start": ..., "target": ..., "feat_static_cat": [0, 0], ... } # red shoes
{ "start": ..., "target": ..., "feat_static_cat": [1, 1], ... } # blue dress
```
在推导时，您可以请求预测其 feat_static_cat 值为在训练数据中观察到的 feat_static_cat 值的组合的目标，例如：

```python
{ "start": ..., "target": ..., "feat_static_cat": [0, 1], ... } # blue shoes
{ "start": ..., "target": ..., "feat_static_cat": [1, 0], ... } # red dress
```

 **2.2 训练数据准则**

 - 时间序列的开始时间和长度可以不同。例如，在营销工作中，产品通常在不同日期进入零售目录，因此，它们的起始日期自然会不同。但是，所有系列必须具有相同的频率、分类特征数量和动态特征数量。
 - 根据文件中时间序列的位置将训练文件随机排序。换而言之，时间序列在文件中以随机顺序出现。
 - 确保正确设置 **start** 字段。算法使用 **start** 时间戳来派生内部特征。
 - 如果您使用分类特征 (**feat_static_cat**)，则所有时间序列必须具有相同数量的分类特征。
 - 如果您的数据集包含 **feat_dynamic_real** 字段，则算法会自动使用该字段。所有时间序列必须具有相同数量的特征时间序列。每个特征时间序列中的时间点与目标中的时间点一一对应。此外，feat_dynamic_real 字段中的条目应具有与 target 相同的长度。如果已使用 **feat_dynamic_real** 字段训练模型，则必须提供此字段以进行推理。此外，每个特征必须具有提供的目标的长度加上 **prediction_length**。

 **2.3 加载路径中包含的JSON Lines文件的数据集**

 **common.FileDataset** 加载路径中包含的 **JSON Lines** 文件的数据集。
 > ***class gluonts.dataset.common.FileDataset(path: pathlib.Path, freq: str, one_dim_target: bool = True)***

 - **path：** 包含数据集文件的路径。每个文件都会被认作训练数据源，除了以"."开头和"_SUCCESS"结尾的文件外。文件中的有效行可以是: {“start”: “2014-09-07”, “target”: [0.1, 0.2]}
 - **freq：** 时间序列中的观测频率
 - **one_dim_target：** 是否仅接受单变量目标时间序列

 **2.4 构造DeepAR网络**
 > ***class gluonts.model.deepar.DeepAREstimator(freq: str, prediction_length: int, trainer: gluonts.trainer._base.Trainer = gluonts.trainer._base.Trainer(batch_size=32, clip_gradient=10.0, ctx=None, epochs=100, hybridize=True, init="xavier", learning_rate=0.001, learning_rate_decay_factor=0.5, minimum_learning_rate=5e-05, num_batches_per_epoch=50, patience=10, weight_decay=1e-08), context_length: Optional[int] = None, num_layers: int = 2, num_cells: int = 40, cell_type: str = 'lstm', dropout_rate: float = 0.1, use_feat_dynamic_real: bool = False, use_feat_static_cat: bool = False, use_feat_static_real: bool = False, cardinality: Optional[List[int]] = None, embedding_dimension: Optional[List[int]] = None, distr_output: gluonts.distribution.distribution_output.DistributionOutput = gluonts.distribution.student_t.StudentTOutput(), scaling: bool = True, lags_seq: Optional[List[int]] = None, time_features: Optional[List[gluonts.time_feature._base.TimeFeature]] = None, num_parallel_samples: int = 100)***

 - **freq：** 时间序列中的观测频率
 - **prediction_length：** 预测范围的长度
 - **context_length：** 在计算预测之前要为RNN展开的步骤数（默认值：None，在这种情况下，context_length = projection_length）
 - **num_layers：** RNN层数（默认值：2）
 - **num_cells：** 每层的RNN信元数（默认值：40）
 - **cell_type：** 要使用的循环单元格类型（可用：“ lstm”或“ gru”；默认值：“ lstm”）
 - **dropout_rate：** 辍学正则化参数（默认值：0.1）
 - **use_feat_dynamic_real：** 是否使用 feat_dynamic_real 数据中的字段（默认值：False）
 - **use_feat_static_cat：** 是否使用 feat_static_cat 数据中的字段（默认值：False）
 - **use_feat_static_real：** 是否使用 feat_static_real 数据中的字段（默认值：False）
 - **cardinality：** 每个分类特征的值数。如果 use_feat_static_cat == True，则必须设置（默认：None）
 - **scaling：** 是否自动缩放目标值（默认值：True）
 - **lags_seq：** 用作RNN输入的滞后目标值的索引（默认值：None，在这种情况下，将根据频率自动确定这些值）
 - **time_features：** 用作RNN输入的时间特征（默认值：None，在这种情况下，它们是根据频率自动确定的）
 - **num_parallel_samples：** 每个时间序列的评估样本数，以在推理期间增加并行度。这是一个不影响准确性的模型优化（默认值：100）

 **2.5 例子**

```python
train_data = common.FileDataset("此处填入训练数据文件夹的绝对路径", freq="H")
test_data  = common.FileDataset("此处填入需要预测数据文件夹的绝对路径", freq="H")

estimator = deepar.DeepAREstimator(
    prediction_length=24,
    context_length=100,
    use_feat_static_cat=True,
    use_feat_dynamic_real=True,
    num_parallel_samples=100,
    cardinality=[2,1],
    freq="H",
    trainer=Trainer(ctx="cpu", epochs=200, learning_rate=1e-3)
)
predictor = estimator.train(training_data=train_data)

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-100:].plot(figsize=(12, 5), linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
plt.legend(["past observations", "median prediction", "90% prediction interval", "50% prediction interval"])
plt.show()

prediction = next(predictor.predict(test_data))
print(prediction.mean)
prediction.plot(output_file='graph.png')
```
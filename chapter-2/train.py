# -*- coding: UTF-8 -*-     

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from gluonts.model import deepar
from gluonts.trainer import Trainer
from gluonts.dataset import common
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor

train_data = common.FileDataset("/home/root/mxnetTS/GluonTS-Learning-in-Action/chapter-2/data/train", freq="H")
test_data  = common.FileDataset("/home/root/mxnetTS/GluonTS-Learning-in-Action/chapter-2/data/val", freq="H")

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

predictor.serialize(Path("/home/root/mxnetTS/GluonTS-Learning-in-Action/chapter-2/model"))
# predictor = Predictor.deserialize(Path("/home/root/mxnetTS/GluonTS-Learning-in-Action/chapter-2/model"))
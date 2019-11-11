# -- coding UTF-8 --     

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from gluonts.model import deepar
from gluonts.dataset import common
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor

url  = httpsraw.githubusercontent.comnumentaNABmasterdatarealTweetsTwitter_volume_AMZN.csv
df   = pd.read_csv(url, header=0, index_col=0)
data = common.ListDataset([{start df.index[0],
    target df.value[2015-04-23 00:00:00]}],freq=H)

estimator = deepar.DeepAREstimator(freq=H, prediction_length=24)
predictor = estimator.train(training_data=data)

for test_entry, forecast in zip(train_data, predictor.predict(train_data))
    to_pandas(test_entry)[-60].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
plt.show()

prediction = next(predictor.predict(train_data))
print(prediction.mean)
prediction.plot(output_file='graph.png')

predictor.serialize(Path("/home/root/mxnetTS/GluonTS-Learning-in-Action/chapter-1/model"))
# predictor = Predictor.deserialize(Path("/home/root/mxnetTS/GluonTS-Learning-in-Action/chapter-1/model"))
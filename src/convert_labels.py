import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open('/project/data/baseline2_bal_downsampled_train.json') as f:
    train = json.load(f)

with open('/project/data/baseline2_bal_downsampled_test.json') as f:
    test = json.load(f)

train, test = {'data':np.asarray(train['data']),'labels':np.asarray(train['labels'])}, {'data':np.asarray(test['data']),'labels':np.asarray(test['labels'])}

train['labels'] = train['labels'].argmax(axis=1)

test['labels'] = test['labels'].argmax(axis=1)

with open('/project/data/baseline2_bal_downsampled_train_alt_label.json', 'w') as f:
    json.dump(train, f, cls=NumpyEncoder)

with open('/project/data/baseline2_bal_downsampled_test_alt_label.json', 'w') as f:
    json.dump(test, f, cls=NumpyEncoder)


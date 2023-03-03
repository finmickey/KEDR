import numpy as np
import os
from data_parser import load_from_tsfile_to_dataframe

def process_data(X):
  tmp = []
  for i in (range(len(X))):
      _x = X.iloc[i, :].copy(deep=True)
      _y = []
      for y in _x:
          if y.isnull().any():
              y = y.interpolate(method='linear', limit_direction='both')
          _y.append(y)
      _y = np.array(_y)
      tmp.append(_y)
  return np.array(tmp, dtype=np.float32)

# Save data as .npy files for faster loading
def saveAndConvert(datasetName):
  # Load data from data/raw/datasetName_TRAIN.ts
  X_train, y_train = load_from_tsfile_to_dataframe('data/raw/' + datasetName + '_TRAIN.ts')
  X_test, y_test = load_from_tsfile_to_dataframe('data/raw/' + datasetName + '_TEST.ts')

  X_train = process_data(X_train)
  X_test = process_data(X_test)

  # Create folder if does not exist
  if not os.path.exists('data/npy'):
    os.makedirs('data/npy')

  # Create folder for specific dataset
  if not os.path.exists('data/npy/' + datasetName):
    os.makedirs('data/npy/' + datasetName)

  # Save data as .npy files
  np.save('data/npy/' + datasetName + '/X_train.npy', X_train)
  np.save('data/npy/' + datasetName + '/y_train.npy', y_train)
  np.save('data/npy/' + datasetName + '/X_test.npy', X_test)
  np.save('data/npy/' + datasetName + '/y_test.npy', y_test)

saveAndConvert("BIDMC32SpO2")
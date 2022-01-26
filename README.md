# Human Activity Recognition with LSTM model (PyTorch)
This project follows the idea from [LSTMs for Human Activity Recognition Time Series Classification](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/) but implementation is done with PyTorch instead of Keras. Moreover, Docker environment and Python notebook are provided for a convenient experiment.

## 1. Installation
### 1.1 Set up environment variables
- Create `.env` and insert these below variables. `UID` is an ID of a current user (check by running a command line `id`). `JUPYTER_CONTAINER_MEM_LIMIT` is used to limit memory usage of Jupyter-lab's container.
```
UID=1000
JUPYTER_CONTAINER_MEM_LIMIT=8g
```

### 1.2 Build Docker image
```
docker-compose build
```

### 1.3 Download the UCI HAR dataset
- Go to https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones or download a file direct from https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
- Extract the file to `data` folder. The folder struture should look like the below
```
❯ ls data
activity_labels.txt  features_info.txt  features.txt  README.txt  test  train
```

### 1.3 Run Docker container
```
docker-compose up -d
```
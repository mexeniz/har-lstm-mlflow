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

## 1.4 Configure Jupyterlab
- Edit `config/jupyter/jupyter_notebook_config.py` by your own.
- For example, set a port of Jupyterlab service by
```
## The port the notebook server will listen on.
c.NotebookApp.port = 8080
```
- set a password hash for Jupyterlab access by running this Python code
```
from notebook.auth import passwd; passwd()
```
After that, place a hash string in the config file at
```
c.NotebookApp.password = 'sha1:place:yourstring'
```

### 1.5 Run Docker container
- Start a container and acces it at `http://localhost:8888`
```
docker-compose up -d
```


# 2. Juppyter Notebooks
# 2.1 Data Exploration
You might want to explore and visualize data. You can do it by using `exploration.ipynb` notebook.

# 2.2 Train and inference
Run code in ``
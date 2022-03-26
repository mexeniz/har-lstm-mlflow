# Human Activity Recognition with LSTM model (PyTorch) and MLFlow Tracking
This project follows the idea from [LSTMs for Human Activity Recognition Time Series Classification](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/) but implementation is done with PyTorch instead of Keras. Moreover, Docker environment and Python notebook are provided for a convenient experiment.

## 1. Installation
### 1.1 Set up environment variables
- Create `.env` and insert these below variables. `UID` is an ID of a current user (check by running a command line `id`). `JUPYTER_CONTAINER_MEM_LIMIT` is used to limit memory usage of Jupyter-lab's container.
```
UID=1000
JUPYTER_CONTAINER_MEM_LIMIT=8g

JUPYTER_EXPOSED_PORT=8888

MYSQL_DATABASE=mlflow
MYSQL_USER=mlflow
MYSQL_PASSWORD=mlflow
MYSQL_ROOT_PASSWORD=mlflow
MYSQL_DATA_PATH=/path/to/har-lstm/data/mlflow_db

MLFLOW_EXPOSED_PORT=5000
MLFLOW_ARTIFACTS_PATH=/path/to/har-lstm/data/mlflow_artifacts
```

### 1.2 Build Docker image
```
docker-compose build
```

### 1.3 Download the UCI HAR dataset and create required directories
- Download a dataset to `./data` folder by running a script `scripts/download_dataset.sh`
```
$ ./scripts/download_dataset.sh ./data
```
Then, you will have a folder `./data/har_dataset` like the below
```
data/har_dataset
├── activity_labels.txt
├── features_info.txt
├── features.txt
├── README.txt
├── test
└── train
```
- Create other required folders for database and MLFLow services
```
$ mkdir -p ./data/mlflow_db ./data/mlflow_artifacts
```

## 1.4 Configure Jupyterlab
- Edit `config/jupyter/jupyter_notebook_config.py` by your own.
- For example, set a set a password hash for Jupyterlab access by running this Python code
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


# 2. Jupyter Notebooks
# 2.1 Data Exploration
You might want to explore and visualize data. You can do it by using `exploration.ipynb` notebook.

# 2.2 Train and inference
Run code in ``
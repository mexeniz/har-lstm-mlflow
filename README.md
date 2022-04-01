# Human Activity Recognition with LSTM model (PyTorch) and MLFlow Tracking
This project follows the idea from [LSTMs for Human Activity Recognition Time Series Classification](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/) but implementation is done with PyTorch instead of Keras. Moreover, Docker environment and Python notebook are provided for a convenient experiment.

<!-- toc -->
- [Human Activity Recognition with LSTM model (PyTorch) and MLFlow Tracking](#human-activity-recognition-with-lstm-model-pytorch-and-mlflow-tracking)
  - [1. Installation](#1-installation)
    - [1.1 Set up environment variables](#11-set-up-environment-variables)
    - [1.2 Build Docker image](#12-build-docker-image)
    - [1.3 Download the UCI HAR dataset and create required directories](#13-download-the-uci-har-dataset-and-create-required-directories)
    - [1.4 Configure Jupyterlab](#14-configure-jupyterlab)
    - [1.5 Install Docker Nvidia runtime](#15-install-docker-nvidia-runtime)
  - [2. Jupyter Notebook Demo](#2-jupyter-notebook-demo)
    - [2.1 Start Docker containers](#21-start-docker-containers)
    - [2.2 Data Exploration](#22-data-exploration)
    - [2.3 Train a model without MLFlow](#23-train-a-model-without-mlflow)
  - [2.4 Train a model with MLFlow](#24-train-a-model-with-mlflow)

<!-- tocstop -->
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
**NOTE:** Please see how to install `docker` at [Get Docker](https://docs.docker.com/get-docker/) and `docker-compose` at [Install Docker Compose](https://docs.docker.com/compose/install/).
Here, I use Docker environment for running 3 services as containers; Jupyterlab, MLFlow tracker service, and SQL database. Docker compose is used for conveniently managing those containers and they are configured by `docker-compose.yaml`. 
Let's build images of each service by
```
$ docker-compose build
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

### 1.4 Configure Jupyterlab
- Edit `config/jupyter/jupyter_notebook_config.py` by your own.
- For example, set a set a password hash for Jupyterlab access by running this Python code
```
from notebook.auth import passwd; passwd()
```
After that, place a hash string in the config file at
```
c.NotebookApp.password = 'sha1:place:yourstring'
```

### 1.5 Install Docker Nvidia runtime
If you want to use Nvidia GPU for training a model, follow the instruction steps at [](). Its runtime dependencies are required by a Docker container to access GPU resources.

**WARNING:** If your machine doesn't have GPU or you don't wanna to use it, you can comment the below lines (config of `har_lstm_jupyterlab` container) in `docker-compose.yaml`
```
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 2. Jupyter Notebook Demo

### 2.1 Start Docker containers
- Start all containers
```
docker-compose up -d
```
- Check if all services are ready
```
docker-compose ps -a
```
You see all containers are up like below
```
❯ docker-compose ps -a
NAME                  COMMAND                  SERVICE             STATUS               PORTS
har_lstm_jupyterlab   "/bin/bash -c 'addus…"   jupyterlab          running              0.0.0.0:8888->8888/tcp, :::8888->8888/tcp
mlflow_db             "/entrypoint.sh mysq…"   mlflow_db           running (starting)   33060/tcp
mlflow_tracker        "bash ./wait-for-it.…"   mlflow              running              0.0.0.0:5000->5000/tcp, :::5000->5000/tcp
```
- After that, you can access Jupyterlab at `http://localhost:8888` (or another port which you've set by `config/jupyter/jupyter_notebook_config.py` section) and MLFlow tracker at `http://localhost:5000`


### 2.2 Data Exploration
You might want to explore and visualize data. You can do it by using `exploration.ipynb` notebook.

### 2.3 Train a model without MLFlow
Run code in `train_lstm.ipynb` from a browser or by quickly running the below command inside the Jupyterlab container. 
```
$ docker exec -it har_lstm_jupyterlab /bin/bash -c "jupyter nbconvert --execute /workspace/train_lstm.ipynb"
```
**NOTE:** You can use `--to notebook` option to output the result notebook which shows train/val loss during traning.
```
$ docker exec -it har_lstm_jupyterlab /bin/bash -c "jupyter nbconvert --execute --to notebook /workspace/train_lstm.ipynb" 
```

## 2.4 Train a model with MLFlow
Run code in `train_lstm_mlflow.ipynb` or by quickly running the below command inside the Jupyterlab container. Unlike `train_lstm.ipynb`, this notebook trains a LSTM model and sends metrics during to MLFlow tracker service.
```
$ docker exec -it har_lstm_jupyterlab /bin/bash -c "jupyter nbconvert --execute /workspace/train_lstm_mlflow.ipynb"
```
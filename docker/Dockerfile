ARG base_image=nvidia/cuda:11.4.2-devel-ubuntu20.04

FROM $base_image

LABEL maintainer="pawissakan.cpcu@gmail.com"
LABEL org.label-schema.description="Jupyter-lab for HAR-LSTM project"

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

WORKDIR /workspace

RUN apt -y update && apt -y upgrade && \ 
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && add-apt-repository -y ppa:linuxuprising/libpng12 && \
    apt -y update && apt install -y \
    sudo python3.9 python3.9-dev python3.9-distutils graphviz curl \
    dirmngr apt-transport-https pkg-config \
    libfreetype6-dev libpng12-0-dev \
    lsb-release ca-certificates && \
    curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash - && \
    apt install -y nodejs gcc g++ make && \
    apt clean -y

RUN rm -f /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python 

# Install pip and required packages
RUN curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas==1.2.0 \
    matplotlib==3.2.1 \
    jupyterlab==3.0.0 \
    jupyterlab-execute-time==2.1.0 \
    scipy==1.7.3 \
    seaborn==0.11.1 \
    altair==4.1.0 \
    ipywidgets==7.5.1 \
    scikit-learn==1.0.1 \
    graphviz==0.14 \
    loguru==0.4.1 \
    torch==1.10.1 \
    numpy==1.22.0

# Install dependencies for altair_saver
RUN npm install -g vega-lite vega-cli canvas

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter lab build

# ENTRYPOINT ["/usr/local/bin/jupyter-lab", "--port=8888", "--ip=0.0.0.0", "--allow-root"]

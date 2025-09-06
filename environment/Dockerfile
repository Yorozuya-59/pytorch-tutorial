FROM ubuntu:22.04

WORKDIR /home/workdir

# Install Python3.10
RUN apt update 
RUN apt upgrade -y
RUN apt install -y python3.10 

# For initial setting of Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 
RUN apt install -y python3-pip 
RUN pip install -U pip 

# Installation of basic libraries
RUN pip install python-dotenv icecream tqdm

# Installation of libraries for data analysis
RUN pip install polars

# Installation of libraries for visualization
RUN pip install matplotlib japanize-matplotlib seaborn

# Installation of libraries for deep learning
RUN pip install torch torcheval torchvision tensorboard

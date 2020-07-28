FROM python:3.8
MAINTAINER Josip Janzic <josip@jjanzic.com>

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*


RUN apt update
RUN apt-get update -y
RUN apt install -y python3 python3-pip


ADD ./ ./

RUN pip3 install -r requirements.txt

CMD ["python3","-u","api.py"]




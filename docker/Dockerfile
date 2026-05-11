FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /workspace

# # Keep logs unbuffered and make matplotlib usable in headless containers.
# ENV PYTHONUNBUFFERED=1 \
#     MPLBACKEND=Agg

RUN apt-get update -y

# install gymnasium
RUN pip install gymnasium[classic_control]
RUN pip install gymnasium[other]

# highway-env
# RUN apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev \
#     libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev \
#     ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
RUN pip install highway-env
RUN pip install stable-baselines3

RUN mkdir -p /deep-learning-from-scratch-4

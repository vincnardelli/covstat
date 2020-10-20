FROM python:3.6
LABEL maintainer="vincnardelli@gmail.com"

# Install the C compiler tools
RUN apt-get update -y && \
  apt-get install build-essential -y && \
  pip install --upgrade pip

# Install packages
RUN pip install numpy pandas matplotlib scipy
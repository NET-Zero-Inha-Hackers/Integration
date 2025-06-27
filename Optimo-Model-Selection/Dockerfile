FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo "$TZ" > /etc/timezone

# set mirror.kakao.com as apt source
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list

# Set work directory
WORKDIR /app

# Install pip and python3
RUN apt-get update && apt-get install -y \
    python3.12 python3-pip \
 && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt .
# pip update to the latest version
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Copy the application code
COPY . /app

# Expose the application port
EXPOSE 8000

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM continuumio/miniconda3:latest

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    lsof \
    vim \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 创建数据目录
RUN mkdir -p /app/data

# 创建日志和运行目录
RUN mkdir -p /app/logs /app/run

# 设置Conda环境
ENV CONDA_DEFAULT_ENV=second-me
ENV PATH /opt/conda/envs/second-me/bin:$PATH

# 创建并激活Conda环境
RUN conda create -n second-me python=3.12 -y && \
    conda init bash && \
    echo "conda activate second-me" >> ~/.bashrc

# 安装Python依赖
RUN /opt/conda/envs/second-me/bin/pip install -e .

# 安装前端依赖 (如果有前端)
WORKDIR /app/lpm_frontend
RUN if [ -f "package.json" ]; then npm install; fi

# 返回主目录
WORKDIR /app

# 暴露后端服务端口
EXPOSE 8002
# 暴露前端服务端口
EXPOSE 3000

# 启动脚本
COPY scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

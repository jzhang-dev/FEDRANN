FROM ubuntu:24.10

USER root
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    time \
    tzdata \
    unzip \
    && apt-get clean \
    && apt-get purge \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt /tmp/
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# 后续所有命令都用虚拟环境
ENV PATH="/opt/venv/bin:$PATH"

# 创建工作目录
WORKDIR /workdir

# 安装 SeqNeighbor
COPY . .
RUN pip install --no-cache-dir .


# 设置默认命令
ENTRYPOINT [ "/workdir/entrypoint.sh" ]
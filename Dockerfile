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



# 安装 SeqNeighbor
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

WORKDIR /workdir
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# 设置默认命令
ENTRYPOINT [ "/app/entrypoint.sh" ]
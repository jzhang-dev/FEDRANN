FROM continuumio/miniconda3:25.3.1-1

RUN apt-get update && apt-get install -y \
    build-essential \
    time \
    tzdata \
    && apt-get clean \
    && apt-get purge \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml --name default && \
    conda clean -afy
RUN conda init bash
ENV PATH=/opt/conda/envs/default/bin:$PATH

# Build kmer_searcher
WORKDIR /app
COPY . .
WORKDIR /app/external/kmer_searcher/
RUN bash ./build.sh
ENV PATH=/app/external/kmer_searcher/build:$PATH

# 安装 SeqNeighbor
WORKDIR /app
RUN pip install --no-cache-dir .

WORKDIR /workdir
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# 设置默认命令
ENTRYPOINT [ "/app/entrypoint.sh" ]
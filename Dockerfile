FROM continuumio/miniconda3:latest

COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml --name default && \
    conda clean -afy
RUN conda init bash
ENV PATH=/opt/conda/envs/default/bin:$PATH

# 安装 SeqNeighbor
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

WORKDIR /workdir
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# 设置默认命令
ENTRYPOINT [ "/app/entrypoint.sh" ]
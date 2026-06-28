FROM nvidia/cuda:13.1.2-cudnn-runtime-ubuntu24.04

LABEL org.opencontainers.image.source=https://github.com/Rose-STL-Lab/PG-BIG
LABEL org.opencontainers.image.description="PG-BIG: personalized guidance for biomechanically informed GenAI"

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libegl1 \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    liblapack3 \
    libblas3 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    && conda config --system --set auto_activate_base false \
    && conda clean -afy

WORKDIR /mnt

COPY env/environment.yaml /mnt/environment.yaml

# mamba (from Miniforge) is faster and uses less peak disk than conda during solves.
RUN mamba env create -f /mnt/environment.yaml && \
    mamba clean --all --yes && \
    conda clean --all --yes && \
    rm -rf /opt/conda/pkgs/* /root/.cache/pip

# Conda links libffi.so.7 -> libffi.so.8; with LD_LIBRARY_PATH pointing at the env,
# system tools (git-remote-https, curl) load that alias and break against libp11-kit.
# Drop the .so.7 names so those tools fall back to the system libffi.so.7.
RUN rm -f ${CONDA_DIR}/envs/pg-big/lib/libffi.so.7 ${CONDA_DIR}/envs/pg-big/lib/libffi.7.so

# Sanity-check core stack (fail the image build if anything is missing).
RUN ${CONDA_DIR}/envs/pg-big/bin/python -c "\
import numpy as np; \
import scipy, sklearn, joblib, torch, torchvision, clip; \
import nimblephysics as nimble; \
import opensim; \
import deepspeed; \
import smplx; \
assert np.__version__.startswith('1.25'), np.__version__; \
assert hasattr(nimble, 'biomechanics'); \
assert hasattr(opensim, 'Model'); \
print('numpy', np.__version__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('opensim', opensim.GetVersion())"

# git must work after conda activate (regression guard for libffi / LD_LIBRARY_PATH).
RUN /bin/bash -lc 'eval "$(conda shell.bash hook)" && conda activate pg-big && \
    /usr/lib/git-core/git-remote-https 2>&1 | grep -q "remote-curl: usage"'

ENV CONDA_DEFAULT_ENV=pg-big
ENV PATH=${CONDA_DIR}/envs/pg-big/bin:${PATH}

SHELL ["conda", "run", "-n", "pg-big", "/bin/bash", "-c"]

CMD ["/bin/bash"]

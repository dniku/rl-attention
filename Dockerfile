FROM tensorflow/tensorflow:1.13.1-gpu-py3 as build
FROM scratch
COPY --from=build / /

ENV LANG="C.UTF-8" \
    PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
    CUDA_VERSION="10.0.130" \
    CUDA_PKG_VERSION="10-0=10.0.130-1" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    NVIDIA_REQUIRE_CUDA="cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"

ARG USERNAME=docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        fish man-db \
        cmake libopenmpi-dev zlib1g-dev \
        libsm6 libxext6 libxrender-dev \
        git ffmpeg \
      && \
    rm -rf /var/lib/apt/lists/ && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml && \
    mkdir /workspace && \
    chown $USERNAME:$USERNAME /workspace
USER $USERNAME:$USERNAME
ENTRYPOINT ["fixuid", "-q"]
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

RUN pip install --user \
        git+https://github.com/RerRayne/stable-baselines@b47fe4b \
        gym[atari]==0.12.0 \
        scikit-image \
        tqdm

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /workspace
CMD ["fish"]

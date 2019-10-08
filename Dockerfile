# Builds a Docker image with the necessary libraries for compiling
# FEniCS. The image is at
# https://hub.docker.com/r/fenicsproject/performance-tests
#
# Authors: Garth N. Wells <gnw20@cam.ac.uk>

ARG PETSC_VERSION=3.12

FROM ubuntu:18.04

WORKDIR /tmp

# Environment variables
ENV OPENBLAS_NUM_THREADS=1

# Non-Python utilities and libraries
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        bison \
        cmake \
        flex \
        g++ \
        gfortran \
        git \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-timer-dev \
        libeigen3-dev \
        liblapack-dev \
        libmpich-dev \
        libopenblas-dev \
        libhdf5-mpich-dev \
        mpich \
        ninja-build \
        python \
        python3-dev \
        pkg-config \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    pip3 install --no-cache-dir setuptools && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git && \
    rm -rf /tmp/*

# Install PETSc from source
ARG PETSC_VERSION
RUN git clone --branch v${PETSC_VERSION} --depth 1 https://gitlab.com/petsc/petsc.git && \
    cd petsc && \
    ./configure --COPTFLAGS="-O2" \
                --CXXOPTFLAGS="-O2" \
                --FOPTFLAGS="-O2" \
                --with-c-support \
                --with-debugging=0 \
                --with-shared-libraries \
                --download-hypre \
                --download-metis \
                --download-parmetis \
                --download-ptscotch \
                --prefix=/usr/local/petsc-32 && \
     make && \
     make install && \
     rm -rf /tmp/*

# By default use the 32-bit build of PETSc
ENV PETSC_DIR=/usr/local/petsc-32

# Build DOLFIN
RUN git clone https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx/cpp && \
    mkdir build && \
    cd ./build && \
    cmake -G Ninja ../ && \
    ninja install && \
    rm -rf /tmp/*
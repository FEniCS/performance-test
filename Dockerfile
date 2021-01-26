# Builds a Docker image with the necessary libraries for compiling
# FEniCS. The image is at
# https://hub.docker.com/r/fenicsproject/performance-tests
#
# Authors: Garth N. Wells <gnw20@cam.ac.uk>

ARG PETSC_VERSION=3.12.4

FROM ubuntu:20.04

WORKDIR /tmp

# Environment variables
ENV OPENBLAS_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

# Non-Python utilities and libraries
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
    -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    bison \
    clang \
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
    python3 \
    python3-dev \
    pkg-config \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN git clone --branch trilinos-release-13-0-1 https://github.com/trilinos/trilinos && \
    mkdir -p trilinos/build && cd trilinos/build && \
    cmake -DTPL_ENABLE_MPI=ON \
    -DMPI_BASE_DIR=/usr/x86_64-linux \
    -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON \
    -DTrilinos_ENABLE_MueLu=ON \
    -DTpetra_INST_INT_LONG=ON \
    -DTpetra_INST_INT_LONG_LONG=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DTrilinos_ENABLE_PyTrilinos=OFF \
    -DTrilinos_ENABLE_Teko=OFF \
    -DTPL_ENABLE_Netcdf=OFF \
    .. && \
    make install && \
    cd ../.. && rm -r trilinos

# Install PETSc from source
ARG PETSC_VERSION
RUN git clone --branch v${PETSC_VERSION} --depth 1 https://gitlab.com/petsc/petsc.git && \
    cd petsc && \
    python3 ./configure --with-64-bit-indices=0 \
    --COPTFLAGS="-O3" \
    --CXXOPTFLAGS="-O3" \
    --FOPTFLAGS="-O3" \
    --with-c-support \
    --with-fortran-bindings=no \
    --with-debugging=0 \
    --with-shared-libraries \
    --download-hypre \
    --download-ptscotch \
    --prefix=/usr/local/petsc-32 && \
    make && \
    make install && \
    git clean -fdx . && \
    python3 ./configure --with-64-bit-indices=1 \
    --COPTFLAGS="-O3" \
    --CXXOPTFLAGS="-O3" \
    --FOPTFLAGS="-O3" \
    --with-c-support \
    --with-fortran-bindings=no \
    --with-debugging=0 \
    --with-shared-libraries \
    --download-hypre \
    --download-ptscotch \
    --prefix=/usr/local/petsc-64 && \
    make && \
    make install && \
    rm -rf /tmp/*

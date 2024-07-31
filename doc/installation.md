# General Instructions
Here's how to install w2dynamics. These instruction should work on any standard desktop machine that feature a Fortran compiler and a lapack implementation in a common location.
## Prerequisites
- cmake > 2.8.5 (REQUIRED)
- Fortran-90 compatible Fortran compiler (REQUIRED)
- C++ compiler that at least exposes the C++11 RNG. (REQUIRED, only trivial fallback provided)
- working MPI implementation(REQUIRED)
- BLAS and LAPACK libraries(REQUIRED)
- Python interpreter > 2.4(REQUIRED)
- FFTW (REQUIRED)
- NFFT(OPTIONAL)
- python-numpy > 1.2 (OPTIONAL)
- libhdf5 > 1.6 with Fortran bindings (OPTIONAL)
- mpi4py (OPTIONAL)
- h5py (OPTIONAL)
- scipy > 0.6 with f2py (OPTIONAL)
- python-configobj (OPTIONAL)

Basically the installer should take care that all the necessary prerequisites are available. The installer will take care of those missing dependencies that are marked as optional. To do that it either uses python-pip or installs it if it's not available. For that magic to work further packages might be required, e.g. a python that can execute pip, ca-certificates, python development libraries(that's python-dev on Debian)and curl or wget.
## Configuration and Build
Having said that the actual configuration and installation step is as follows:
In the w2dynamics source directory you do:
- mkdir build
- cd build
- cmake ..
- make CTQMC

to create CTQMC. Or

- mkdir build
- cd build
- cmake ..
- make MAXENT

to create the maxent binary. NFFT is fetched if missing, if other libraries are missing you just get an error indicating that they are missing. We now get the pip installer in case some relevant python libraries are missing. After compilation of the "all" target you can run make test to run the testsuite with make test.

## Special steps for using the PGI toolchain
W2Dynamics can compile natively with the PGI toolchain. To make sure that F2PY is able to find 
the PGI compilers please make sure that the following environment variables are set(PGI defaults for version 17.10 are given):

- export PGI=/opt/pgi
- export PATH=/opt/pgi/linux86-64/17.10/bin:$PATH

Then we have to set cmake up the proper way so that it's using the PGI compiler and hence we end up with
- mkdir build
- cmake -DCMAKE_Fortran_COMPILER=pgfortran -DCMAKE_CXX_COMPILER=pgc++ ..
- make CTQMC

This will link using the standard system lapack libraries

# Instructions for building on MacOS
Here we collect the installation instructions for various alternative package managers on MacOS.
in any case you need the prerequisites mentioned in the section above. There is no need to 
install an additional Lapack and Blas Library since MacOS already brings its own via the Accelerate framework.

## Installation using brew.
We can report successful builds using the brew package manager with the following packages:
- cmake version 3.7.2
- GNU Fortran (Homebrew GCC 7.1.0) 7.1.0
- python: stable 2.7.13 (bottled), HEAD
- numpy: stable 1.13.0 (bottled), HEAD
- scipy: stable 0.19.0 (bottled), HEAD
- open-mpi: stable 2.1.1 (bottled), HEAD
- homebrew/science/hdf5: stable 1.10.1 (bottled)
- python-configobj probably installed via pip.

## Installation using macports
We have observed succesful builds using the following:

- sudo port install cmake gcc fftw3 py27-scipy py27-h5py py27-mpi4py py27-configobj

# Instructions for building on the HCLM
## gfortran and OpenBLAS
In the w2dynamics source directory you do:
- mkdir build
- cd build
- cmake -DPYTHON_LIBRARY=/opt/python/lib/libpython2.7.so -DPYTHON_INCLUDE_DIR=/opt/python/include/python2.7/ -DPYTHON_EXECUTABLE=/opt/python/bin/python -DCMAKE_Fortran_COMPILER=/usr/local/bin/gfortran -DBLAS_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so -DLAPACK_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so ..
- make

to build all source files.
Note that the cmake line above may also be used by people using Anaconda to point cmake to their $ANACONDA_HOME by replacing /opt/python with $ANACONDA_HOME .

## ifort and OpenBLAS(ifort currently not working(Aug. 2017))
In the w2dynamics source directory you do:
- mkdir build
- cd build
- cmake -DPYTHON_LIBRARY=/opt/python/lib/libpython2.7.so -DPYTHON_INCLUDE_DIR=/opt/python/include/python2.7/ -DPYTHON_EXECUTABLE=/opt/python/bin/python -DCMAKE_Fortran_COMPILER=ifort -DBLAS_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so -DLAPACK_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so ..
- make

## ifort + MKL(ifort currently not working(Aug. 2017))
First you need to source the relevant mkl environment variables: source /opt/intel/mkl/bin/mklvars.sh intel64
- mkdir build
- cd build
- cmake -DPYTHON_LIBRARY=/opt/python/lib/libpython2.7.so -DPYTHON_INCLUDE_DIR=/opt/python/include/python2.7/ -DPYTHON_EXECUTABLE=/opt/python/bin/python -DCMAKE_Fortran_COMPILER=ifort -DBLAS_LIBRARIES=-lmkl_rt -DLAPACK_LIBRARIES=-lmkl_rt ..
- make 

# Instructions for building in Würzburg.
Executing the default installation instructions yields a binary compiled with ifort 10 and linked against the system lapack implementation(ATLAS). Specific installation instructions follow
## gfortran + default BLAS implementation
The instructions to compile with gfortran and ATLAS are as follows.

In the w2dynamics source directory you do:

- mkdir build
- cd build
- cmake -DCMAKE_Fortran_COMPILER=gfortran ..
- make

## ifort + MKL
- source /opt/intel/mkl/10.1.0.015/tools/environment/mklvarsem64t.sh 
- mkdir build
- cd build
- cmake -DCMAKE_Fortran_COMPILER=ifort -DBLAS_LIBRARIES="-lmkl_em64t -liomp5 -lpthread -lm" -DLAPACK_LIBRARIES=-lmkl_em64t ..
- make
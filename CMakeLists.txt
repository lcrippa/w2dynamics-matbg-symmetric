# CMake project file for ctqmc_fortran

##################################################
# Define the project and the dependencies that it has
##################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(ctqmc Fortran CXX C)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif(POLICY CMP0074)

# Set the version
SET(VERSION 1.0)

# Set the module path, so that CMake also considers our modules
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Uncomment if it is required that Fortran 90 is supported
IF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    MESSAGE(FATAL_ERROR "Fortran compiler does not support F90")
ENDIF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)

# Set some options the user may choose
# Uncomment the below if you want the user to choose a parallelization library
OPTION(USE_MPI "Use the MPI library for parallelization" ON)
#OPTION(USE_OPENMP "Use OpenMP for parallelization" ON)

# This INCLUDE statement executes code that sets the compile flags for DEBUG,
# RELEASE, and TESTING.  You should  review this file and make sure the flags 
# are to your liking.
option(OPT_HOST_ARCH "Target host machine instruction set" ON)
INCLUDE(${CMAKE_MODULE_PATH}/SetFortranFlags.cmake)
# Locate and set parallelization libraries.  There are some CMake peculiarities
# taken care of here, such as the fact that the FindOpenMP routine doesn't know
# about Fortran.
INCLUDE(${CMAKE_MODULE_PATH}/SetParallelizationLibrary.cmake)
# Setup the LAPACK libraries.  This also takes care of peculiarities, such as
# the fact the searching for MKL requires a C compiler, and that the results
# are not stored in the cache. 
INCLUDE(${CMAKE_MODULE_PATH}/SetUpLAPACK.cmake)

# There is an error in CMAKE with this flag for pgf90.  Unset it
GET_FILENAME_COMPONENT(FCNAME ${CMAKE_Fortran_COMPILER} NAME)
IF(FCNAME STREQUAL "pgf90")
    UNSET(CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS)
ENDIF(FCNAME STREQUAL "pgf90")

SET( CMAKE_Fortran_FLAGS             "${CMAKE_Fortran_FLAGS} -DLAPACK77_Interface" )

############################################################
# Define the actual files and folders that make up the build
############################################################

find_package(PythonInterp 3.0)
if (NOT PYTHONINTERP_FOUND)
  find_package(PythonInterp 2.6)
endif()
find_package(PackageHandleStandardArgs)


INCLUDE(${CMAKE_MODULE_PATH}/installviapip.cmake)

option(USE_NFFT "Compile with support for routines requiring the NFFT library for non-equispaced FFTs" ON)

if (USE_NFFT)
  find_package(NFFT "3.3.0")
  if(NOT NFFT_FOUND)
    find_package(FFTW REQUIRED)
    list(GET FFTW_LIBRARIES 0 FFTW_LIBFILEPATH)
    get_filename_component(FFTW_LIBDIR "${FFTW_LIBFILEPATH}" DIRECTORY)
    include(ExternalProject)
    set(_nfft_version "3.5.3")
    ExternalProject_Add(nfft_local
      URL https://www-user.tu-chemnitz.de/~potts/nfft/download/nfft-${_nfft_version}.tar.gz
      PREFIX nfft_local
      CONFIGURE_COMMAND <SOURCE_DIR>/configure $<$<NOT:$<BOOL:${OPT_HOST_ARCH}>>:--enable-portable-binary> --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib --with-fftw3-libdir=${FFTW_LIBDIR} --with-fftw3-includedir=${FFTW_INCLUDES}
      BUILD_COMMAND make
      INSTALL_COMMAND make install
      URL_HASH SHA256=caf1b3b3e5bf8c33a6bfd7eca811d954efce896605ecfd0144d47d0bebdf4371
      )
    # in the previous findPackage(NFFT) call an old version of NFFT could have been found. Here we override it with our up-to-date self-compiled version. We don't use the include directories.
    set(NFFT_LIBRARIES ${PROJECT_BINARY_DIR}/nfft_local/lib/libnfft3.a)
    add_library(nfft STATIC IMPORTED)
    set_target_properties(nfft PROPERTIES IMPORTED_LOCATION ${NFFT_LIBRARIES})
    add_dependencies(nfft nfft_local)
  endif(NOT NFFT_FOUND)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DUSE_NFFT=1")
else ()
  set(NFFT_LIBRARIES "")
endif ()

find_package(NUMPY 1.10)
IF(NOT NUMPY_FOUND)
    install_via_pip("numpy>=1.10.0" ERRORCODE)
    if (${ERRORCODE})
    MESSAGE(FATAL_ERROR "Couldn't install numpy -> ABORTING!!!")
    endif()
ENDIF()


find_package(H5PY)

find_package(MPI4PY)
find_package(SCIPY 0.10)
IF(NOT SCIPY_FOUND)
    install_via_pip("scipy" ERRORCODE)
    if (${ERRORCODE})
    MESSAGE(FATAL_ERROR "Couldn't install scipy -> ABORTING!!!")
    endif()
ENDIF()
find_package(CONFIGOBJ)
if(NOT H5PY_FOUND)
    install_via_pip("h5py" ERRORCODE)
    if (${ERRORCODE})
    MESSAGE(FATAL_ERROR "Couldn't install H5PY -> ABORTING!!!")
    endif()
endif()
if(NUMPY_VERSION VERSION_GREATER "1.17.99" AND SCIPY_VERSION VERSION_LESS "1.0.0")
    # automatic installation of h5py via pip can also pull in quite
    # new versions of numpy as dependencies...
    install_via_pip("scipy>=1.0.0" ERRORCODE)
    if (${ERRORCODE})
      message(WARNING "Your versions of numpy and scipy are incompatible and updating scipy automatically failed. You will not be able to run w2dynamics before updating scipy.")
    endif()
endif()
if(NOT MPI4PY_FOUND AND MPI_Fortran_FOUND)
    install_via_pip("mpi4py" ERRORCODE)
    if (${ERRORCODE})
    MESSAGE(FATAL_ERROR "Couldn't install MPI4PY -> ABORTING!!!")
    endif()
endif()
if(NOT CONFIGOBJ_FOUND)
    install_via_pip("configobj" ERRORCODE)
    if (${ERRORCODE})
    MESSAGE(FATAL_ERROR "Couldn't install configobj(required for DMFT.py) -> ABORTING!!!")
    endif()
endif()
#set up some common F2PY Variables
INCLUDE(${CMAKE_MODULE_PATH}/SetupF2Py.cmake)

# Define the executable name
SET(CTQMCEXE ctqmc)

# Define some directories
SET(SRC ${CMAKE_SOURCE_DIR}/src)
SET(LIB ${CMAKE_CURRENT_BINARY_DIR}/lib)
#SET(BIN ${CMAKE_SOURCE_DIR}/bin)

SET(SRCCTQMC ${SRC}/ctqmc_fortran)
SET(SRCMAXENT ${SRC}/maxent)
SET(SRCMTRNG ${SRC}/mtrng)

# Have the .mod files placed in the lib folder
SET(CMAKE_Fortran_MODULE_DIRECTORY ${LIB})

# The source for the C++11 MTRNG and have it placed in the lib folder
ADD_SUBDIRECTORY(${SRCMTRNG} ${LIB})

# The source for the ctqmc binary and have it placed in the bin folder
ADD_SUBDIRECTORY(${SRCCTQMC})
ADD_SUBDIRECTORY(${SRCMAXENT})

# Add a distclean target to the Makefile
ADD_CUSTOM_TARGET(distclean 
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/distclean.cmake
)

ADD_SUBDIRECTORY(testsuite/mtrng.tests)
ADD_SUBDIRECTORY(testsuite/maxent.tests)
ADD_SUBDIRECTORY(testsuite/ctqmc.tests)
ADD_SUBDIRECTORY(testsuite/w2dyn.tests)
enable_testing()


# INSTALLATION

message(STATUS "Configuration for installation (optional):")

# default / fallback paths relative to CMAKE_INSTALL_PREFIX
set(INSTALL_PATH_SITE_PACKAGES "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
set(INSTALL_PATH_SCRIPTS "bin")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'));"
    RESULT_VARIABLE PY_SYSCONFIG_PLATLIB_FAILURE
    OUTPUT_VARIABLE PY_SYSCONFIG_PLATLIB_PATH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (NOT PY_SYSCONFIG_PLATLIB_FAILURE)
    message(STATUS "Using path obtained from Python sysconfig module for package installation")
    message(STATUS "Package installation path: ${PY_SYSCONFIG_PLATLIB_PATH}")
    set(INSTALL_PATH_SITE_PACKAGES "${PY_SYSCONFIG_PLATLIB_PATH}")
  endif()

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('scripts'));"
    RESULT_VARIABLE PY_SYSCONFIG_SCRIPTS_FAILURE
    OUTPUT_VARIABLE PY_SYSCONFIG_SCRIPTS_PATH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (NOT PY_SYSCONFIG_SCRIPTS_FAILURE)
    message(STATUS "Using path obtained from Python sysconfig module for script installation")
    message(STATUS "Script installation path: ${PY_SYSCONFIG_SCRIPTS_PATH}")
    set(INSTALL_PATH_SCRIPTS "${PY_SYSCONFIG_SCRIPTS_PATH}")
  endif()

  if ((NOT PY_SYSCONFIG_PLATLIB_FAILURE) OR (NOT PY_SYSCONFIG_SCRIPTS_FAILURE))
    message(STATUS "Set CMAKE_INSTALL_PREFIX to override default installation paths")
  endif()

else()

  message(STATUS "Installation prefix directory (variable CMAKE_INSTALL_PREFIX): ${CMAKE_INSTALL_PREFIX}")

endif()

option(INSTALL_EXECUTABLES "Install executable scripts DMFT.py, cthyb, Maxent.py and hgrep to the executable directory (depends on prefix, e.g. /usr/bin)" ON)

message(STATUS "Installing executable scripts DMFT.py, cthyb, Maxent.py, hgrep (option INSTALL_EXECUTABLES): ${INSTALL_EXECUTABLES}")

# Compile all Python files to bytecode
install(SCRIPT PyCompileall.txt)

install(DIRECTORY w2dyn
  DESTINATION "${INSTALL_PATH_SITE_PACKAGES}"
  COMPONENT Python_Package)
if(INSTALL_EXECUTABLES)
  set(SCRIPT_FILES DMFT.py cthyb hgrep Maxent.py)
  foreach(SCRIPT_FILE ${SCRIPT_FILES})
    install(
      CODE  "file(READ \"${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT_FILE}\" SCRIPT_CONTENT)
             file(WRITE \"${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT_FILE}.cmake.out\" \"#!${PYTHON_EXECUTABLE}\n# Preceding path inserted by cmake during installation\n\")
             file(APPEND \"${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT_FILE}.cmake.out\" \"\${SCRIPT_CONTENT}\")"
      COMPONENT Executable_Scripts
    )
    install(
      PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT_FILE}.cmake.out"
      DESTINATION "${INSTALL_PATH_SCRIPTS}"
      COMPONENT Executable_Scripts
      RENAME "${SCRIPT_FILE}"
    )
    install(CODE  "file(REMOVE \"${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT_FILE}.cmake.out\")"
            COMPONENT Executable_Scripts)
  endforeach()
endif()

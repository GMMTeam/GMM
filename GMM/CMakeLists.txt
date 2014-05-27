CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT (GMMing)

SET (USE_CUDA 0)
SET (BUILD_GMM_TRAIN 1)
SET (BUILD_GMM_LOGLIKE 1)

# Boost available from: http://www.boost.org/users/download/
FIND_PACKAGE(Boost REQUIRED)

# if FIND_PACKAGE not able to locate the Boost dirs, comment the previous line and
# you can try to specify the paths to boost include and lib directories manually,
# uncomment and adjust following three lines:
#SET (BOOST_ROOT "/usr/boost")
#SET (Boost_INCLUDE_DIRS "${BOOST_ROOT}/boost_1_39_0")
#SET (Boost_LIBRARY_DIRS "${BOOST_ROOT}/lib64bit")

# in some builds following boost-libs has to be specified (uncomment if necessary)
SET (Boost_MY_LIB_FILES
#	"boost_filesystem-mt"
#	"boost_system-mt"
#	"boost_program_options-mt"
#	"boost_thread-mt"
)

# ACML available from: http://developer.amd.com/tools-and-sdks/cpu-development/amd-core-math-library-acml/acml-downloads-resources/
SET (ACML_ROOT_DIR "/usr/acml5.3.1")
SET (ACML_INCLUDE_DIR "${ACML_ROOT_DIR}/ifort64/include")
SET (ACML_LIB_DIR "${ACML_ROOT_DIR}/ifort64/lib")
SET (ACML_MY_LIB_FILES "acml")

IF (${USE_CUDA})
	FIND_PACKAGE(CUDA REQUIRED)
	# name of additional .lib file(s), depends on the version of CUDA
	SET (CUDA_ADDLIB_FILE "cuda")
	
	# if necessary additional CUDA .lib files can be added 
	# (needed in some of the CUDA versions)
	# LIST (APPEND CUDA_ADDLIB_FILE "cudart")

	# set types of GPU architectures to be build for
	SET (CUDA_NVCC_FLAGS "-gencode=arch=compute_11,code=compute_11")
	LIST (APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=compute_20")
ENDIF (${USE_CUDA})

add_subdirectory (lib_bmssr)
add_subdirectory (projects)

INCLUDE_DIRECTORIES (
		"${Boost_INCLUDE_DIRS}" 
		"${ACML_INCLUDE_DIR}" 
		"${GMMing_SOURCE_DIR}/lib_bmssr"
)

ADD_DEFINITIONS (
	-D_ACML
	-DEXCEPTION_SUPPORT
	-D_EXCLUDE_SAFETY_CONDS_
)

IF (${USE_CUDA})
	INCLUDE_DIRECTORIES (
		"${CUDA_INCLUDE_DIRS}"
	)
	ADD_DEFINITIONS (
		-D_CUDA
	)
	CUDA_ADD_LIBRARY (trainerCUDA
		CU_GMMStatsEstimator_CUDA.cu
		CU_GMMStatsEstimator_CUDAkernels.cuh
		OPTIONS -D_CUDA
	)
ENDIF (${USE_CUDA})

ADD_LIBRARY (trainer
	OL_ModelAdapt.cpp 
	OL_GMMStatsEstimator.cpp
	OL_GMMStatsEstimator_SSE.cpp
	OL_GMMStatsEstimator_CUDA.cpp
)


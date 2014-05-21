INCLUDE_DIRECTORIES (
		"${ACML_INCLUDE_DIR}"
		"${Boost_INCLUDE_DIRS}" 
		"${GMMing_SOURCE_DIR}/lib_bmssr"
)

LINK_DIRECTORIES (
		"${ACML_LIB_DIR}"
		"${Boost_LIBRARY_DIRS}" 
		"${GMMing_BINARY_DIR}/lib_bmssr/general"
		"${GMMing_BINARY_DIR}/lib_bmssr/matrices"
		"${GMMing_BINARY_DIR}/lib_bmssr/model"
		"${GMMing_BINARY_DIR}/lib_bmssr/param"
		"${GMMing_BINARY_DIR}/lib_bmssr/tools"
		"${GMMing_BINARY_DIR}/lib_bmssr/trainer"
)

ADD_DEFINITIONS (
	-D_ACML
	-D_EXCLUDE_SAFETY_CONDS_
)

IF (${USE_CUDA})	
	INCLUDE_DIRECTORIES (
		"${CUDA_INCLUDE_DIRS}"
	)
	LINK_DIRECTORIES (
		"${CUDA_LIBRARIES}"
	)
	ADD_DEFINITIONS (
		-D_CUDA
	)
	SET (CUDA_LINK_DEPS trainerCUDA ${CUDA_ADDLIB_FILE})
ENDIF (${USE_CUDA})

ADD_EXECUTABLE (getLogLike main.cpp) 
TARGET_LINK_LIBRARIES (getLogLike 
	trainer
	${CUDA_LINK_DEPS}
	model 
	param
	generalX 
	tools 
	matrices 
	${ACML_MY_LIB_FILES}
	${Boost_MY_LIB_FILES}
) 

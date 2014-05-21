INCLUDE_DIRECTORIES (
		"${Boost_INCLUDE_DIRS}" 
		"${GMMing_SOURCE_DIR}/lib_bmssr"
)
ADD_LIBRARY(tools
	OL_FileList.cpp
	OL_MLFParser.cpp
	usefull_funcs.h
)

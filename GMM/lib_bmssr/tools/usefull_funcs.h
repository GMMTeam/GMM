// 
// Copyright (c) 2007--2014 Lukas Machlica
// Copyright (c) 2007--2014 Jan Vanek
// 
// University of West Bohemia, Department of Cybernetics, 
// Plzen, Czech Repulic
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

#ifdef __GNUC__
#include "general/my_inttypes.h"
#endif

#ifndef MY_MAX_LINE_LENGTH
#define MY_MAX_LINE_LENGTH  1000
#endif

#include "general/GlobalDefine.h"
#include "tools/OL_FileList.h"

#include <string>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <boost/filesystem.hpp>

namespace utilities {

	// get the last dash position + 1 -> position after last dash
	inline const char *getBasename(const char *path);

	// replace dashes -> '\' -> '/'	
	inline void replace_dashes(std::string *filename);

	// IN: full path 'filepath', desired output path in 'newpath' and desired output extension in 'outExt'
	// OUT: 'out' - 'filepath's path will be replaced by 'newpath', basename of the file will be kept 
	//      and its extension will be replaced by 'outExt'
	// note: all of the '\' dashes will be replaced by '/'
	inline void GetOutPath(char *out, const char *filepath, const char *newpath, const char *outExt);
	inline void GetNewPath(std::string &out, std::string &filepath, 
						   std::string &newpath, std::string &outExt);
	// dashes will be no replaced (beware of '\' dashes! - use '/' instead of)
	inline void GetNewPathC(std::string &out, const std::string &filepath, 
							const std::string &newpath, const std::string &outExt);	


	// IN: 'fileOrDirname' .. path + name of file or directory
	//     'mask' .. has the form '.ext', if mask=NULL => mask=".*"
	//
	// OUT: testlist -> list of files contained in direcory or file specified by 'fileOrDirname' and 'mask';
	//      note: 'testlist' has to be pre-allocated 
	inline void ReadTestList(const char *fileOrDirname, CFileList *testlist, 
					const char *mask = NULL, bool FilesFromSubdirs = false);	

	// load MLLR or fMLLR matrix, 'size_x' & 'size_y' will be filled in this function
	template <typename T>
	inline T **loadMatrix(const char *filename, unsigned int &size_x, unsigned int &size_y);
	
	// IN: 'pathToList' .. path+name of list with N lines given as:
	//					   GROUP_ID_i file_i1,file_i2,...,file_iM  
	//					   note: only one space is allowed on each line separating group_ID 
	//					   and list of files; following files are separated only by comma!
	//     'additional_path' .. prefix added to 'groupLists' before each file
	//     'additional_ext' .. suffix added to 'groupLists' after each file
	// OUT: 'groupNames' .. {GROUP_ID_i,...,GROUP_ID_N}
	//      'groupLists' .. return value; {file_11,...,file_1M},...,{file_N1,...,file_NM}
	// note: 'groupNames' have to be allocated before the call
	inline CFileList **LoadMergingLists(std::string &pathToList, CFileList &groupNames, 
									    std::string &additional_path, std::string &additional_ext);
	
	// IN: 'unsorted_array' .. array of unsorted numbers
	//     'dim' .. dimension of unsorted array
	//     'N' .. how many indexes should be returned (N <= dim)
	// OUT: array containing sequence of indexes denoting sorted elements in the 'unsorted_array'
	template <typename T>
	inline unsigned int *getSortedIDXs(T *unsorted_array, unsigned int dim, unsigned int N);
	template <typename T>
	void getSortedIDXs(unsigned int *sorted_idxs, T **p_unsorted_array, T *unsorted_array, 
					   unsigned int dim, unsigned int N);

	// ---------------------------------------------------------------------
	// FUNC DEFS -----------------------------------------------------------	
	// ---------------------------------------------------------------------
	const char *getBasename(const char *path) {
		const char *last_dash = path, *dash = path-1;		
		while((dash = strchr(dash+1, '\\')) != NULL)
			last_dash = dash+1;
	
		dash = last_dash-1;
		while((dash = strchr(dash+1, '/')) != NULL)
			last_dash = dash+1;

		return last_dash;
	}


	template <typename T>
	int Compare(T *element1, T *element2) {
		if(*element1 <= *element2)
			return 0;
		return 1;
	}

	template <typename T>
	unsigned int *getSortedIDXs(T *unsorted_array, unsigned int dim, unsigned int N) {
		if(N > dim)
			N = dim;
		
		unsigned int *sorted_idxs;
		if((sorted_idxs = new unsigned int [N]) == NULL)
			throw std::runtime_error("sortIDXs(): Not enough memory!\n\t");

		T **p_unsorted_array;
		if((p_unsorted_array = new T* [dim]) == NULL)
			throw std::runtime_error("sortIDXs(): Not enough memory!\n\t");

		for(unsigned int i = 0; i < dim; i++)
			p_unsorted_array[i] = &(unsorted_array[i]);

		getSortedIDXs(sorted_idxs, p_unsorted_array, unsorted_array, dim, N);

		delete p_unsorted_array;
		return sorted_idxs;
	}

	template <typename T>
	void getSortedIDXs(unsigned int *sorted_idxs, 
		  			   T **p_unsorted_array, T *unsorted_array, 
					   unsigned int dim, unsigned int N) 
	{
		if(N > dim)
			N = dim;

		std::sort<T **>(p_unsorted_array, p_unsorted_array + dim, Compare<T>);
						
		for(unsigned int i = 0; i < N; i++)
			sorted_idxs[i] = (unsigned int) (p_unsorted_array[i] - unsorted_array);						
	}

	void GetOutPath(char *OUTname_DEST, const char *INpathName, const char *OUTpath, const char *outExt) {
		std::string m(INpathName);	

		// find slash -> '\' or '/' (iterating from the end of the std::string)
		size_t slash_location = m.rfind("/");
		if (slash_location == std::string::npos) {
			slash_location = m.rfind("\\");
			if (slash_location == std::string::npos)
				slash_location = 0;
			else slash_location += 1;
		}
		else slash_location += 1;

		// find dot (iterating from the end of the std::string)
		size_t dot_location = m.rfind(".");
		if (dot_location == std::string::npos)
			dot_location = m.length()+1;		

		std::string path(OUTpath);
		if(path.length())
			path += "/";	

		// get the model out path
		std::string basename = path + m.substr(slash_location,dot_location-slash_location) + outExt;	
		// replace dashes -> '\' -> '/'	
		replace_dashes(&basename);
		
		strcpy(OUTname_DEST, basename.c_str());
	}

	void replace_dashes(std::string *filename) {
		size_t f = 0;
		while((f = filename->find("\\", f)) != std::string::npos)
			filename->replace(f, 1, "/");	
	}

	void GetNewPath(std::string &out, std::string &filepath, std::string &newpath, std::string &outExt) {
		replace_dashes(&filepath);
		replace_dashes(&newpath);
		GetNewPathC(out, filepath, newpath, outExt);
	}

	void GetNewPathC(std::string &out, const std::string &filepath, 
					 const std::string &newpath, const std::string &outExt) {	
		
		out.clear();		
		size_t slash_location = filepath.rfind("/");
		if (slash_location == std::string::npos)
			slash_location = 0;
		else
			++slash_location;

		if(newpath.size() > 0)
			out += newpath + "/" + filepath.substr(slash_location, filepath.size());
		else
			out += filepath.substr(slash_location, filepath.size());

		if(outExt.compare(".*") != 0) {			
			size_t slash_location = out.rfind("/");
			size_t dot_location = out.rfind(".");
			if (dot_location == std::string::npos || dot_location < slash_location)
				out += outExt;
			else
				out.replace(dot_location, out.length()-dot_location, outExt);					
		}
	}

	void ReadTestList(const char *fileOrDirname, CFileList *testlist, const char *mask /*= NULL*/, bool FilesFromSubdirs /*= false*/) 
	{				
		if ( !boost::filesystem::exists(fileOrDirname) )
			return;

		boost::filesystem::path path(fileOrDirname);
		if ( !boost::filesystem::is_directory( path )  )
		{
			std::ifstream file(fileOrDirname);	
			if(!file.fail()) {
				int linenum = 0;
				char line[MY_MAX_LINE_LENGTH];
				while(file.getline(line,MY_MAX_LINE_LENGTH)) {
					if(line[0] == '\n')
						continue;

					if(line[strlen(line)-1] == '\r')
						line[strlen(line)-1] = '\0';
					testlist->AddItem(line);
					++linenum;
				}
				file.close();
				//if(linenum == 0) 
				//	throw std::runtime_error("\n\tReadTestList(): Non files have been found in " << filename << "!", true);
			}
		}
		else {
			if(mask == NULL) {
				if(testlist->ReadDir(fileOrDirname, "*.*", FilesFromSubdirs) == 1) {				
					std::string errors = std::string("ReadTestList(): Reading of directory \"") + fileOrDirname + "\" content failed!";
					throw std::runtime_error(errors.c_str());
				}
			}
			else {
				std::string m("*");
				m.append(mask);
				if(testlist->ReadDir(fileOrDirname, m.c_str(), FilesFromSubdirs) == 1) {
					std::string errors = std::string("ReadTestList(): Reading of directory \"") + fileOrDirname + "\" content failed!";
					throw std::runtime_error(errors.c_str());
				}
			}
		}
	}


	template <typename T>
	T **loadMatrix(const char *filename,unsigned int &size_x, unsigned int &size_y) {

		std::ifstream file(filename);	
		if(file.fail())
			return NULL;

		char line[MY_MAX_LINE_LENGTH];
		file.getline(line, MY_MAX_LINE_LENGTH); // line - matrix ID: fMLLR
		if(strcmp(line, " fMLLR") != 0 && strcmp(line, " MLLR") != 0) {
			std::string errors = std::string("loadMatrix():Unknown format -> ") + filename + "!";
			throw std::runtime_error(errors.c_str());
		}


		file.getline(line, MY_MAX_LINE_LENGTH); // line: matrix_num size_x size_y	
		unsigned int shift, mnum;
		if(sscanf(line, "%d %d %d", &mnum, &size_x, &size_y) == 0) {
			std::string errors = std::string("loadMatrix():Unknown format -> ") + filename + "!";
			throw std::runtime_error(errors.c_str());
		}

		T **mat;
		mat = new T* [size_x];
		for(unsigned int i = 0; i < size_x; i++) {
			mat[i] = new T [size_y];
		}
		char num[50];
		for(unsigned int i = 0; i < size_x; i++) {
			file.getline(line, MY_MAX_LINE_LENGTH);
			shift = 0;
			for(unsigned int j = 0; j < size_y; j++) {
				sscanf(line + shift, "%s", num);
				shift += strlen(num) + 1;
				mat[i][j] = (T) atof(num);
			}
		}
		file.close();	
		return mat;
	}
		
	CFileList **LoadMergingLists(std::string &pathToList, CFileList &groupNames, 
								 std::string &additional_path, std::string &additional_ext) 
	{
		std::string newFilename;
		std::ifstream file(pathToList.c_str());
		if(file.fail()) {
			std::string errors = std::string("tLoadMergingLists():List ") + pathToList + " not found!";
			throw std::runtime_error(errors.c_str());
		}

		char line[MY_MAX_LINE_LENGTH];
		unsigned int linenum = 0;
		while(file.getline(line, MY_MAX_LINE_LENGTH)) {
			if(line[0] != '\n')
				linenum++;
		}

		CFileList **groupLists;
		groupLists = new CFileList* [linenum];		

		file.clear();
		file.seekg(0, std::ios::beg);
		char c, group_id[MY_MAX_LINE_LENGTH], foo[MY_MAX_LINE_LENGTH];
		for(unsigned int i = 0; i < linenum; i++) {
			file.getline(line, MY_MAX_LINE_LENGTH);
			if(line[strlen(line)-1] == '\r')
				line[strlen(line)-1] = '\0';

			sscanf(line, "%s %s", group_id, foo);

			groupNames.AddItem(group_id);
			groupLists[i] = new CFileList(group_id);			

			unsigned int j = strlen(group_id)+1, idx = 0;
			do {			
				c = line[j++];
				if(c == ',' || c == '\0') {
					foo[idx] = '\0';
					std::string foo2(foo);
					GetNewPath(newFilename, foo2, additional_path, additional_ext);			
					replace_dashes(&newFilename);
					groupLists[i]->AddItem(newFilename.c_str());
					idx = 0;
				}
				else if(c != ' ') 
					foo[idx++] = c;
			} while(c != '\0');
		}

		file.close();
		return groupLists;
	}	
}
namespace bmssr = utilities; // pre spetnu kompatibilitu s inymi projektami - do buducnosti odstranit!

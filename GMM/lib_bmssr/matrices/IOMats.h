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

#ifndef _LoadMats_
#define _LoadMats_

#include <iostream>
#include <fstream>
#include <string>

// bool include_head -- write/read the size of the matrix to/from the file?
// if false => when loading => x,y assumed to be 1
template <typename T>
void loadMat (const char *filename, bool txt,
			  T*& mat, unsigned int& x, unsigned int& y, bool include_head = true);

template <typename T>
void loadMatTxt (std::ifstream& file, T*& mat, unsigned int& x, unsigned int& y, bool include_head = true);

template <typename T>
void loadMatBin (std::ifstream& file, T*& mat, unsigned int& x, unsigned int& y, bool include_head = true);

// ---------------------------------------------

template <typename T>
void saveMat (const char *filename, bool txt,
			  T* mat, unsigned int x, unsigned int y, bool include_head = true);

template <typename T>
void saveMatTxt (std::ofstream& file, T* mat, unsigned int x, unsigned int y, bool include_head = true);

template <typename T>
void saveMatBin (std::ofstream& file, T* mat, unsigned int x, unsigned int y, bool include_head = true);

// ---------------------------------------------

template <typename T>
void printMat (T *mat, unsigned int x, unsigned int y);




// ---------------------------------------------
// ---------------------------------------------

template <typename T>
void printMat (T *mat, unsigned int x, unsigned int y) {

	std::cout << "[" << x << "," << y << "]" << std::endl;
	for(unsigned int i = 0; i < x; i++) {
		for(unsigned int j = 0; j < y; j++) {
			std::cout << mat[i*y + j] << " ";
		}
		std::cout << std::endl;
	}
}



template <typename T>
void loadMat (const char *filename, bool txt,
			  T*& mat, unsigned int& x, unsigned int& y, bool include_head) 
{
	std::ifstream file;
	if(txt)
		file.open(filename);
	else
		file.open(filename, std::ios::binary);

	if(file.fail()) {
		std::string error_report = std::string("loadMat(): Unable to open file ") + filename;
		throw std::runtime_error(error_report.c_str());
	}

	file.exceptions (std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);

	try {
		if(txt)
			loadMatTxt(file, mat, x, y, include_head);
		else
			loadMatBin(file, mat, x, y, include_head);
	}
	catch (std::ifstream::failure& e) {
		//std::cout << "loadMat(): Exception reading file" << std::endl << e.what() << std::endl;
		std::string error_report = std::string("loadMat(): Exception reading file\n") + e.what();
		throw std::runtime_error(error_report.c_str());
	}

	
	file.close();
}



template <typename T>
void loadMatTxt (std::ifstream& file, T*& mat, unsigned int& x, unsigned int& y, bool include_head) 
{
	std::string line;
	getline(file, line);		
	
	if (include_head)
		sscanf(line.c_str(), "%d %d", &x, &y);
	else {
		x = y = 1;
	}
	mat = new T [x*y];
	
	for(unsigned int i = 0; i < x; i++) {
		getline(file, line);

		std::stringstream lineparser(line); 
		for(unsigned int j = 0; j < y; j++)
			lineparser >> mat[i*y + j];
	}
}



template <typename T>
void loadMatBin (std::ifstream& file, T*& mat, unsigned int& x, unsigned int& y, bool include_head) 
{
	if (include_head) {
		__int32 __x;
		file.read(reinterpret_cast<char *> (&__x), sizeof(__int32));
		x = static_cast<unsigned int> (__x);

		file.read(reinterpret_cast<char *> (&__x), sizeof(__int32));
		y = static_cast<unsigned int> (__x);
	}
	else {
		x = y = 1;
	}
	
	mat = new T [x*y];
	file.read(reinterpret_cast<char *> (mat), sizeof(T) * x * y);
}



template <typename T>
void saveMat (const char *filename, bool txt,
			  T* mat, unsigned int x, unsigned int y, bool include_head)
{	
	std::ofstream file;
	if(txt)
		file.open(filename, std::ios::out | std::ios::trunc);
	else
		file.open(filename, std::ios::out | std::ios::trunc | std::ios::binary);
	
	if(file.fail()) {
		std::string error_report = std::string("saveMat(): file ") + filename + " could not be opened for writing";
		throw std::runtime_error(error_report.c_str());
	}

	file.exceptions(std::ofstream::eofbit | std::ofstream::failbit | std::ofstream::badbit);

	try {
		if(txt)
			saveMatTxt(file, mat, x, y, include_head);
		else
			saveMatBin(file, mat, x, y, include_head);
	}
	catch (std::ofstream::failure& e) {
		//std::cout << "saveMat(): Exception writing file" << std::endl << e.what() << std::endl;
		std::string error_report = std::string("saveMat(): Exception writing file\n") + e.what();
		throw std::runtime_error(error_report.c_str());
	}


	file.close();
}



template <typename T>
void saveMatTxt (std::ofstream& file, T* mat, unsigned int x, unsigned int y, bool include_head)
{
	if (include_head)
		file << x << " " << y << std::endl;

	for(unsigned int i = 0; i < x; i++) {
		for(unsigned int j = 0; j < y; j++) {
			file << mat[i*y + j] << " ";
		}
		file << "\n";
	}
	file << std::endl;
}



template <typename T>
void saveMatBin (std::ofstream& file, T* mat, unsigned int x, unsigned int y, bool include_head)
{
	if (include_head) {
		__int32 __x = static_cast<__int32> (x);
		file.write(reinterpret_cast<char *> (&x), sizeof(__int32));

		__int32 __y = static_cast<__int32> (y);
		file.write(reinterpret_cast<char *> (&y), sizeof(__int32));
	}

	file.write(reinterpret_cast<char *> (mat), sizeof(T) * x * y);
}


#endif

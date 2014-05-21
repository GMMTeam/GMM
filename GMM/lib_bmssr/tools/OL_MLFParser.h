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

#ifndef _OL_MLF_PARSER_
#define _OL_MLF_PARSER_

#include <vector>
#include <string>
#include <list>
#ifdef __GNUC__
#	include <ext/hash_map>
#define stdext __gnu_cxx
#else
#	include <hash_map>
#endif


class MLF_Parser {
public:

	// 2 dimensional vector [0] = frame begin, [1] = frame end
	typedef std::vector<unsigned int> element_range;
	
	// htkfilename X -> list of frame ranges of given class contained int X
	typedef stdext::hash_map <std::string, std::list <element_range> > file_2_ranges; 


	static void parseClasses (const char* filename, std::list <std::string>& lmodels, 
							  stdext::hash_map <std::string, std::list <std::string> >& elem2class);
	
	static void parseMLF (const char* filename, 
						  stdext::hash_map <std::string, std::list <std::string> >& elem2class, 
						  stdext::hash_map <std::string, file_2_ranges>& class2frames);
	
	static void print (const char* filename, stdext::hash_map <std::string, file_2_ranges>& class2frames);
};

#endif

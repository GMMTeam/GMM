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

#include "OL_MLFParser.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>

#ifdef __GNUC__
namespace __gnu_cxx
{
        template<> struct hash< std::string >
        {
                size_t operator()( const std::string& x ) const
                {
                        return hash< const char* >()( x.c_str() );
                }
        };
}
#endif

using namespace std;
namespace fs = boost::filesystem;
using namespace stdext;

void MLF_Parser::parseClasses (const char* filename, list <string>& lmodels, 
							   hash_map <string, list <string> >& elem2class) 
{
	ifstream file(filename);
	if(file.fail()) {
		string error_report = string("parseClasses(): file ") + filename + " could not be opened for reading";
		throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::badbit);
	try {
		string line, item, classname;
		while(getline(file, line)) {
			istringstream lineparser(line);
			lineparser >> classname;
			while (lineparser >> item) {
				elem2class[item].push_back(classname);
				item.clear();
			}
			lmodels.push_back(classname);
		}
	}
	catch (ifstream::failure& f) {		
		file.close();
		
		string error_report = string("parseClasses(): Error occured when reading file ") + filename + "\n\t" + f.what();
		throw runtime_error(error_report.c_str());
	}	

	file.close();	
}


namespace {
	void fixPath (const string& pathTobeFixed, string& new_path) 
	{
		new_path.clear();

		unsigned int njump = 0;
		string::const_iterator it = pathTobeFixed.begin();
		while (*it == '"' || *it == ' ' || *it == '*' || *it == '/' || *it == '\\') {
			it++;
			njump++;
		}

		size_t dash_pos = pathTobeFixed.find_last_of('/');
		if (dash_pos == string::npos)
			dash_pos = pathTobeFixed.find_last_of('\\');

		if (dash_pos != string::npos && dash_pos > njump) {
			new_path.assign(it, pathTobeFixed.begin() + dash_pos);
			new_path += '/';
		}

		new_path += fs::basename(fs::path(pathTobeFixed));
	}
}

void MLF_Parser::parseMLF (const char* filename, hash_map <string, list <string> >& elem2class, 
						   hash_map <string, file_2_ranges >& class2frames)
{
	ifstream file(filename);
	if(file.fail()) {
		string error_report = string("MLF_Parser::parse(): file ") + filename + " could not be opened for reading";
		throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::badbit);
	try {
		string line, htkfile, element;
		vector<unsigned int> er;		

		er.resize(2);
		while(getline(file, line)) {
			if (line.empty() || line.at(0) == '\r' || line.at(0) == '.' || line.at(0) == '#')
				continue;
			
			if (line.at(0) == '"') {
				fixPath (line, htkfile);
				continue;
			}
			istringstream lineparser(line);
			lineparser >> er[0] >> er[1] >> element;

			if (elem2class.find(element) != elem2class.end()) {			
				list <string>::iterator it;
				for (it = elem2class[element].begin(); it != elem2class[element].end(); it++) {
					file_2_ranges& f2r = class2frames[*it];
					list <element_range> &ER = f2r[htkfile];
					ER.push_back(er);
					//class2frames[*it][htkfile].push_back(er);
				}
			}
		}	
	}
	catch (ifstream::failure& f) {		
		file.close();
		
		string error_report = string("MLF_Parser::parse(): Error occured when reading file ") + filename + "\n\t" + f.what();
		throw runtime_error(error_report.c_str());
	}	

	file.close();
}




void MLF_Parser::print (const char* filename, hash_map <string, file_2_ranges >& class2frames) {
	ofstream ofile(filename);
	if(ofile.fail()) {
		string error_report = string("print(): file ") + filename + " could not be opened for reading";
		throw runtime_error(error_report.c_str());
	}

	ofile.exceptions(ofstream::badbit);
	try {		
		hash_map <string, hash_map < string, list < vector <unsigned int> > > >::iterator it;
		for (it = class2frames.begin(); it != class2frames.end(); it++) {
			ofile << "[" << (*it).first << "]\n";

			// HTKFILES + frame intervals
			file_2_ranges& file2ranges = class2frames[(*it).first];
			for (file_2_ranges::iterator it1 = file2ranges.begin(); it1 != file2ranges.end(); it1++) {
				// htkfile
				ofile << "\t" << (*it1).first << " -> ";

				list < vector <unsigned int> >::iterator it2;
				for (it2 = file2ranges[(*it1).first].begin(); it2 != file2ranges[(*it1).first].end(); it2++) {
					// frames interval
					ofile << (*it2).at(0) << "-" << (*it2).at(1) << ";";
				}
				ofile << "\n";
			}

		}
	}
	catch (ifstream::failure& f) {		
		ofile.close();
		
		string error_report = string("print(): Error occured when writing to file ") + filename + ".\n" + f.what();
		throw runtime_error(error_report.c_str());
	}	

	ofile.close();
}

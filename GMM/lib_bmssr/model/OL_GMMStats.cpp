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

#ifdef _OL_GMM_STATS_H_

#ifdef __GNUC__
#	include "general/my_inttypes.h"
#endif

#include <new>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <assert.h>



template <typename T>
GMMStats<T>::GMMStats() {	
	initializeVariables();
}



template <typename T>
GMMStats<T>::GMMStats (unsigned int modeldim, unsigned int nummix,
					   bool allocMeans /*= true*/, bool allocVars /*= false*/, 
					   bool allocFullVars /*= false*/, bool allocAux /*= false*/) 
{
	initializeVariables();
	alloc(modeldim, nummix, allocMeans, allocVars, allocFullVars, allocAux);
	reset();
}



template <typename T>
GMMStats<T>::GMMStats (const GMMStats<T>& gmmStats) {
	clone(gmmStats);
}



template <typename T>
GMMStats<T>::~GMMStats() {	
	deleteStats();
}



template <typename T>
GMMStats<T>& GMMStats<T>::operator= (const GMMStats<T>& gmmStats) {
	if(&gmmStats != this)				
		clone(gmmStats);

	return *this;
}



template <typename T>
const GMMStats<T> GMMStats<T>::operator+ (const GMMStats<T>& rgmmStats) const {	
	GMMStats<T> gmmStats(*this);
	gmmStats += rgmmStats;
	return gmmStats;
}



template <typename T>
GMMStats<T>& GMMStats<T>::operator+= (const GMMStats<T>& rgmmStats) {
	
	if(_ms.size() != rgmmStats._ms.size())
		throw std::logic_error("operator+= : Inconsistent statistics differing in number of mixtures!\n\t");
	if(_allM && rgmmStats._allM && _dim != rgmmStats._dim)
		throw std::logic_error("operator+= : Inconsistent statistics differing in dimensionality!\n\t");
	// note: _allV need not to be checked as it is derived from _dim

	for(unsigned int i = 0; i < _ms.size(); i++) 
	{		
		_ms[i].mixProb += rgmmStats._ms[i].mixProb;
		unsigned int shift = 0;
		for(unsigned int n = 0; n < _dim; n++) 
		{
			if(_allM && rgmmStats._allM)		
				_ms[i].mean[n] += rgmmStats._ms[i].mean[n];

			if(_allA && rgmmStats._allA) {
				_ms[i].aux[n] += rgmmStats._ms[i].aux[n];
				_ms[i].aux2 += rgmmStats._ms[i].aux2;
			}

			if(_allV && rgmmStats._allV) {
				shift += n;
				if(_allFV && rgmmStats._allFV) {					
					for(unsigned int nn = n; nn < _dim; nn++)
						_ms[i].var[n * _dim + nn - shift] += rgmmStats._ms[i].var[n * _dim + nn - shift];
				}
				else if(!_allFV && rgmmStats._allFV) {
					_ms[i].var[n] += rgmmStats._ms[i].var[n * _dim + n - shift];
				}
				else if (_allFV && !rgmmStats._allFV) {
					_ms[i].var[n * _dim + n - shift] += rgmmStats._ms[i].var[n];
				}
				else _ms[i].var[n] += rgmmStats._ms[i].var[n];
			}	
		}
	}
	_totalAccSamples += rgmmStats._totalAccSamples;
	_totLogLike += rgmmStats._totLogLike;

	return *this;
}



template <typename T>
typename GMMStats<T>::mixStats& GMMStats<T>::operator[] (unsigned int mixidx) {
	if (mixidx >= _ms.size()) 
		throw std::out_of_range("operator[]: Specified index out of bounds! \n\t");		

	return _ms[mixidx];
}



template <typename T>
void GMMStats<T>::initializeVariables() {
	_allocated = false;	
	_allM = false;
	_allV = false;
	_allFV = false;
	_allA = false;
	_totalAccSamples = 0;
	_totLogLike = 0.0;
	_varsize = 0;
	_dim = 0;	
}



template <typename T>
void GMMStats<T>::clone (const GMMStats& gmmStats) {	
	
	if(gmmStats.isAllocated()) {
		if(!_allocated) {
			alloc(gmmStats);
		}
		else if(_dim != gmmStats._dim || _ms.size() != gmmStats._ms.size() ||
			_allM != gmmStats._allM || _allV != gmmStats._allV || _allFV != gmmStats._allFV || _allA != gmmStats._allA)
		{
			deleteStats();
			alloc(gmmStats);
		}

		for (unsigned int i = 0; i < _ms.size(); i++)
			gmmStats.getMixStats(_ms[i], i, false);

		_totalAccSamples = gmmStats._totalAccSamples;
		_totLogLike = gmmStats._totLogLike;
	}
	else deleteStats();
}



template <typename T>
void GMMStats<T>::alloc (const GMMStats& gmmStats) {	
	alloc(gmmStats._dim, gmmStats._ms.size(), 
		  gmmStats._allM, gmmStats._allV, gmmStats._allFV, gmmStats._allA);
}



template <typename T>
void GMMStats<T>::alloc (unsigned int modeldim, unsigned int nummix,
						 bool allocMeans /*= true*/, bool allocVars /*= true*/, 
						 bool allocFullVars /*= false*/, bool allocAux /*= false*/)
{	
	if (_allocated && ( (modeldim != _dim) || (nummix != _ms.size()) || 
		(allocMeans != _allM) || (allocVars != _allV) || 
		(allocFullVars != _allFV) || (allocAux != _allA) ) )
	{
		deleteStats();
		//throw std::logic_error("alloc(): Allocation already performed!\n\t");		
	}	

	// Pre-alloc space for statistics of each GMM mixture.
	_ms.reserve(nummix);

	_allM = allocMeans;
	_allV = allocVars;
	_allFV = allocFullVars;
	_allA = allocAux;
		
	//if (_allFV)	{
	//	_allV = _allM = true;
	//	_varsize = (modeldim * (modeldim + 1)) / 2;
	//}
	//else if(_allV) {
	//	_allM = true;
	//	_varsize = modeldim;		
	//}
	if (_allFV)	{
		_allV = true;
		_varsize = (modeldim * (modeldim + 1)) / 2;
	}
	else if(_allV) {
		_varsize = modeldim;		
	}

	_dim = modeldim;
		
	// Variable _allocated need to be set here in order to
	// perform actions in deleteStats() if the allocation fails.
	_allocated = true;

	for (unsigned int i = 0; i < nummix; i++) {	
		_ms.push_back(mixStats());		
		allocMixStats(_ms[i], _allM, _dim, _allV, _varsize, _allA, _dim);
	}	
}



template <typename T>
void GMMStats<T>::allocMixStats (mixStats& ms, bool allocMeans, unsigned int mdim,
								 bool allocVars, unsigned int vdim,
								 bool allocAux, unsigned int adim) const {
	if (allocMeans) 
		ms.mean = new T[mdim]; // T's ctor can also throw an exception!
	else ms.mean = NULL;

	if (allocAux) 
		ms.aux = new T[adim]; // T's ctor can also throw an exception!
	else ms.aux = NULL;

	if (allocVars)
		ms.var = new T[vdim]; // T's ctor can also throw an exception!
	else ms.var = NULL;
}



template <typename T>
void GMMStats<T>::deleteStats() {
	deleteStats(0, _ms.size());
}



template <typename T>
void GMMStats<T>::deleteStats (unsigned int begin, unsigned int end) {
	assert(begin <= end && end <= _ms.size());

	if (_allocated) {
		if (_allM) {
			for (unsigned int it = begin; it < end; it++)
				delete [] _ms[it].mean;
		}
		if (_allV) {
			for (unsigned int it = begin; it < end; it++)
				delete [] _ms[it].var;
		}
		if (_allA) {
			for (unsigned int it = begin; it < end; it++)
				delete [] _ms[it].aux;
		}
		_ms.clear();
	}
					
	initializeVariables();
}



template <typename T>
void GMMStats<T>::reset() {

	typename std::vector<mixStats>::iterator it;
	for (it = _ms.begin(); it != _ms.end(); it++) {
		(*it).mixProb = 0;
		if (_allocated) {
			for (unsigned int i = 0; i < _dim; i++) {
				if( _allM)
					(*it).mean[i] = 0;
				if (_allV)
					(*it).var[i] = 0;
				if( _allA) {
					(*it).aux[i] = 0;
					(*it).aux2 = 0;
				}
			}			
			if (_allFV) {
				for(unsigned int i = _dim; i < _varsize; i++)
					(*it).var[i] = 0;
			}
		}
		else {
			(*it).mean = NULL;
			(*it).var = NULL;
			(*it).aux = NULL;
		}
	}
	_totalAccSamples = 0;
	_totLogLike = 0.0;
}



template <typename T>
void GMMStats<T>::getMixStats (mixStats& msDest, unsigned int mixidx, 
							   bool allocate /*= true*/) const {
	if (mixidx >= _ms.size()) 
		throw std::out_of_range("getMixStats(): Specified index out of bounds! \n\t");		
	
	if (allocate)
		allocMixStats(msDest, _allM, _dim, _allV, _allA, _varsize);	

	// Memcpy is not used - what if T would be a class?
	// Note: 'operator =' has to be defined for class T.
	if (_allM) {
		for (unsigned int i = 0; i < _dim; i++)
			msDest.mean[i] = _ms[mixidx].mean[i];
	}
	
	if (_allV) {
		for (unsigned int i = 0; i < _varsize; i++)
			msDest.var[i] = _ms[mixidx].var[i];
	}

	if (_allA) {
		msDest.aux2 = _ms[mixidx].aux2;
		for (unsigned int i = 0; i < _dim; i++)
			msDest.aux[i] = _ms[mixidx].aux[i];
	}

	msDest.mixProb = _ms[mixidx].mixProb;
}



template <typename T>
inline bool GMMStats<T>::isAllocated() const {
	return _allocated;
}



template <typename T>
inline bool GMMStats<T>::getAllocStatus (bool& allocMeans, bool& allocVars, 
										 bool& allocFullVars, bool& allocAux) const
{
	allocMeans = _allM;
	allocVars = _allV;
	allocFullVars = _allFV;
	allocAux = _allA;
}



template <typename T>
inline bool GMMStats<T>::hasFullCov() const {
	return _allFV;
}



template <typename T>
inline unsigned int GMMStats<T>::getTotAccSamples() const {
	return _totalAccSamples;
}



template <typename T>
inline T GMMStats<T>::getTotLogLike() const {
	return _totLogLike;
}



template <typename T>
inline unsigned int GMMStats<T>::getDim() const {
	return _dim;
}



template <typename T>
inline unsigned int GMMStats<T>::getVarDim() const {
	return _varsize;
}



template <typename T>
inline unsigned int GMMStats<T>::getMixNum() const {
	return static_cast<unsigned int> (_ms.size());
	//return _ms.size();
}



template <typename T>
inline T GMMStats<T>::getProb (unsigned int mixidx) const {
	if (mixidx >= _ms.size()) 
		throw std::out_of_range("getProb(): Specified index out of bounds! \n\t");				
	
	return _ms[mixidx].mixProb;
}



template <typename T>
inline const T* GMMStats<T>::getMean (unsigned int mixidx) const {
	if (mixidx >= _ms.size()) 
		throw std::out_of_range("getMean(): Specified index out of bounds! \n\t");		
	
	return _ms[mixidx].mean;
}



template <typename T>
inline const T* GMMStats<T>::getVar (unsigned int mixidx) const {
	if (mixidx >= _ms.size()) 
		throw std::out_of_range("getVar(): Specified index out of bounds! \n\t");		
	
	return _ms[mixidx].var;
}



template <typename T>
inline const T* GMMStats<T>::getAux (unsigned int mixidx) const {
	if (mixidx >= _ms.size()) 
		throw std::out_of_range("getVar(): Specified index out of bounds! \n\t");		
	
	return _ms[mixidx].aux;
}



template <typename T>
inline const T GMMStats<T>::getAux2 (unsigned int mixidx) const {
	if (mixidx >= _ms.size()) 
		throw std::out_of_range("getVar(): Specified index out of bounds! \n\t");		
	
	return _ms[mixidx].aux2;
}



template <typename T>
void GMMStats<T>::load (const char* filename, bool loadTXT, 
						bool loadMeans, bool loadVars) 
{
	std::ifstream ifile;

	if(loadTXT)	
		ifile.open(filename);
	else
		ifile.open(filename, std::ios::binary);

	if(ifile.fail()) {
		std::string error_report = std::string("load(): Unable to open file ") + filename;
		throw std::runtime_error(error_report.c_str());
	}

	ifile.exceptions (std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);

	try {
		if(loadTXT)	
			loadTxt(ifile, loadMeans, loadVars);
		else
			loadBin(ifile, loadMeans, loadVars);
	}
	catch (std::ifstream::failure& e) {
		// std::cout << "load(): Exception reading file" << std::endl << e.what() << std::endl;
		std::string error_report = std::string("save(): Exception reading file\n") + e.what();
		throw std::runtime_error(error_report.c_str());		
	}

	ifile.close();
}



template <typename T>
void GMMStats<T>::loadBin (std::ifstream& ifile, bool loadMeans, bool loadVars) 
{
	bool mprobSaved, meanSaved, varSaved, fullVarSaved;
	unsigned int dim, nummix, varsize;

	__int32 foo;
	ifile.read(reinterpret_cast<char *> (&foo), sizeof(__int32));
	mprobSaved = static_cast<bool> (foo != 0);

	ifile.read(reinterpret_cast<char *> (&foo), sizeof(__int32));
	meanSaved = static_cast<bool> (foo != 0);

	ifile.read(reinterpret_cast<char *> (&foo), sizeof(__int32));
	varSaved = static_cast<bool> (foo != 0);

	ifile.read(reinterpret_cast<char *> (&foo), sizeof(__int32));
	fullVarSaved = static_cast<bool> (foo != 0);

	ifile.read(reinterpret_cast<char *> (&foo), sizeof(__int32));
	dim = static_cast<unsigned int> (foo);

	ifile.read(reinterpret_cast<char *> (&foo), sizeof(__int32));
	varsize = static_cast<unsigned int> (foo);

	ifile.read(reinterpret_cast<char *> (&foo), sizeof(__int32));
	nummix = static_cast<unsigned int> (foo);

	if(!_allocated) {
		alloc (dim, nummix, loadMeans & meanSaved, loadVars & varSaved, loadVars & fullVarSaved);
	}
	else if(_dim != dim || _ms.size() != nummix ||
		    _allM != (loadMeans & meanSaved) || 
			_allV != (loadVars & varSaved) || 
			_allFV != (loadVars &fullVarSaved)) 		
	{
		deleteStats();
		alloc (dim, nummix, loadMeans & meanSaved, loadVars & varSaved, loadVars & fullVarSaved);
	}
	
	for (unsigned int m = 0; m < nummix; m++) 
	{
		if(mprobSaved)
			ifile.read(reinterpret_cast<char *> (&_ms[m].mixProb), sizeof(T));
		
		if(meanSaved) {
			if(loadMeans) 
				ifile.read(reinterpret_cast<char *> (_ms[m].mean), sizeof(T) * dim);
			else
				ifile.seekg(sizeof(T) * dim, std::ios_base::cur);

		}
		if(varSaved) {				
			if(loadVars)
				ifile.read(reinterpret_cast<char *> (_ms[m].var), sizeof(T) * varsize);
			else
				ifile.seekg(sizeof(T) * varsize, std::ios_base::cur);
		}	
	} // for m
}



template <typename T>
void GMMStats<T>::loadTxt (std::ifstream& ifile, bool loadMeans, bool loadVars) 
{
	std::string line;
	getline (ifile, line);

	std::stringstream fileParser(line);
	fileParser.str(line);

	bool mprobSaved, meanSaved, varSaved, fullVarSaved;
	unsigned int nummix, dim, varsize;

	fileParser >> mprobSaved >> meanSaved >> varSaved >> fullVarSaved >> dim >> varsize >> nummix;
	getline (ifile, line);

	fileParser.clear();
	fileParser.seekg(std::ios_base::beg);	

	if(!_allocated) {
		alloc (dim, nummix, loadMeans & meanSaved, loadVars & varSaved, loadVars & fullVarSaved);
	}
	else if(_dim != dim || _ms.size() != nummix ||
		    _allM != (loadMeans & meanSaved) || 
			_allV != (loadVars & varSaved) || 
			_allFV != (loadVars &fullVarSaved)) 		
	{
		deleteStats();
		alloc (dim, nummix, loadMeans & meanSaved, loadVars & varSaved, loadVars & fullVarSaved);
	}	

	for (unsigned int m = 0; m < nummix; m++) 
	{
		if(mprobSaved) {				
			getline (ifile, line);
			fileParser.str(line);
			fileParser >> _ms[m].mixProb;
			fileParser.clear();
			fileParser.seekg(std::ios_base::beg);
		}
		if(meanSaved) {
			getline (ifile, line);
			if(loadMeans) {
				fileParser.str(line);
				for (unsigned int i = 0; i < dim; i++)
					fileParser >> _ms[m].mean[i];
				fileParser.clear();
				fileParser.seekg(std::ios_base::beg);
			}

		}
		if(varSaved) {
			getline (ifile, line);
			if(loadVars) {
				fileParser.str(line);
				for (unsigned int i = 0; i < varsize; i++)
					fileParser >> _ms[m].var[i];
				fileParser.clear();
				fileParser.seekg(std::ios_base::beg);
			}
		}	
		getline (ifile, line);
	} // for m
}



template <typename T>
void GMMStats<T>::save (const char* filename, bool saveTXT, 
						bool saveMProbs, bool saveMeans, 
						bool saveVars, bool saveAuxS) const
{
	std::ofstream ofile;

	if(saveTXT)	
		ofile.open(filename, std::ios::out | std::ios::trunc);
	else
		ofile.open(filename, std::ios::out | std::ios::trunc | std::ios::binary);

	if(ofile.fail()) {
		std::string error_report = std::string("save(): Unable to open file ") + filename;
		throw std::runtime_error(error_report.c_str());
	}

	ofile.exceptions(std::ofstream::eofbit | std::ofstream::failbit | std::ofstream::badbit);

	try {
		if(saveTXT)	
			saveTxt(ofile, saveMProbs, saveMeans, saveVars);
		else
			saveBin(ofile, saveMProbs, saveMeans, saveVars);
	}
	catch (std::ofstream::failure& e) {
		// std::cout << "save(): Exception writing file" << std::endl << e.what() << std::endl;
		std::string error_report = std::string("save(): Exception writing file\n") + e.what();
		throw std::runtime_error(error_report.c_str());
	}


	ofile.close();

	if (saveAuxS && _allA) {
		std::string outAux = std::string(filename) + ".aux";
		saveAux(outAux.c_str());
	}
}



template <typename T>
void GMMStats<T>::saveTxt (std::ofstream& ofile, bool saveMProbs,
						   bool saveMeans, bool saveVars) const
{
	bool sMeans = saveMeans & _allM;
	bool sVars = saveVars & _allV;

	unsigned int nummix = _ms.size();

	ofile << saveMProbs << " " << sMeans << " " << sVars << " " << (_allFV & sVars) << " ";
	ofile << _dim << " " << _varsize << " " << nummix << std::endl << std::endl;
	for (unsigned int m = 0; m < nummix; m++)
	{
		if(saveMProbs)
			ofile << _ms[m].mixProb << std::endl;

		if(sMeans) {
			for (unsigned int i = 0; i < _dim; i++)
				ofile << " " << _ms[m].mean[i];
			ofile << std::endl;
		}

		if(sVars) {
			for (unsigned int i = 0; i < _varsize; i++)
				ofile << " " << _ms[m].var[i];
			ofile << std::endl;
		}

		ofile << std::endl;
	}
}



template <typename T>
void GMMStats<T>::saveBin (std::ofstream& ofile, bool saveMProbs,
						   bool saveMeans, bool saveVars) const
{
	bool sMeans = saveMeans & _allM;
	bool sVars = saveVars & _allV;

	unsigned int nummix = _ms.size();

	__int32 foo;
	foo = static_cast<__int32> (saveMProbs);
	ofile.write(reinterpret_cast<char *> (&foo), sizeof(__int32));	

	foo = static_cast<__int32> (sMeans);
	ofile.write(reinterpret_cast<char *> (&foo), sizeof(__int32));
	
	foo = static_cast<__int32> (sVars);
	ofile.write(reinterpret_cast<char *> (&foo), sizeof(__int32));
	
	foo = static_cast<__int32> (sVars & _allFV);
	ofile.write(reinterpret_cast<char *> (&foo), sizeof(__int32));	

	foo = static_cast<__int32> (_dim);
	ofile.write(reinterpret_cast<char *> (&foo), sizeof(__int32));
	
	foo = static_cast<__int32> (_varsize);
	ofile.write(reinterpret_cast<char *> (&foo), sizeof(__int32));

	foo = static_cast<__int32> (nummix);
	ofile.write(reinterpret_cast<char *> (&foo), sizeof(__int32));
	
	for (unsigned int m = 0; m < nummix; m++)
	{
		if(saveMProbs) {
			T mp = _ms[m].mixProb;
			ofile.write(reinterpret_cast<char *> (&mp), sizeof(T));
		}

		if(sMeans)
			ofile.write(reinterpret_cast<char *> (_ms[m].mean), sizeof(T) * _dim);

		if(sVars)
			ofile.write(reinterpret_cast<char *> (_ms[m].var), sizeof(T) * _varsize);
	}
}


template <typename T>
void GMMStats<T>::saveAux (const char* filename) const
{
	std::ofstream ofile;
	ofile.open(filename, std::ios::out | std::ios::trunc);

	if(ofile.fail()) {
		std::string error_report = std::string("saveAux(): Unable to open file ") + filename;
		throw std::runtime_error(error_report.c_str());
	}

	ofile.exceptions(std::ofstream::eofbit | std::ofstream::failbit | std::ofstream::badbit);

	try {
		unsigned int nummix = _ms.size();
		ofile << _dim << " " << nummix << std::endl << std::endl;
		if (_allA) {
			for (unsigned int m = 0; m < nummix; m++)
			{			
				ofile << _ms[m].aux2 << std::endl;
				for (unsigned int i = 0; i < _dim; i++)
					ofile << " " << _ms[m].aux[i];
				ofile << std::endl;			
			}
		}
	}
	catch (std::ofstream::failure& e) {
		// std::cout << "save(): Exception writing file" << std::endl << e.what() << std::endl;
		std::string error_report = std::string("saveAux(): Exception writing file\n") + e.what();
		throw std::runtime_error(error_report.c_str());
	}


	ofile.close();
}
#endif

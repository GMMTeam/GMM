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

#ifndef _KW_PARAMETRIZATION_
#define _KW_PARAMETRIZATION_

#ifdef __GNUC__
#include "general/my_inttypes.h"
#endif

#include <string.h>
#include <stdexcept>
#include <vector>
#include <list>

class CParamKW;


#define SVES_PRM_IN 0
#define HTK_PRM_IN 1
#define RAW_PRM_IN 2

// UKLADANIE DAT:
//		. alokuje sa jeden velky blok pre vsetky data (NSamples * Dim) a pole 
//		  pointrov '**Vectors', ktore ukazuje na casti tohto bloku predstavujuce 
//		  jednotlive vektory
//		. v pripade pripojenia novych dat (viz funkcia ExportInternal()), sa alokuje
//		  dalsi blok o velikosti NSamples_novych_dat * Dim, vytvori sa nove pole
//		  pointrov '**Vectors', ktore jedna cast <0,NSamples) ukazuje na prvy blok
//        a cast <NSamples, NSamples + NSamples_novych_dat) na druhy/novy blok
class Param {
public:

	Param();
	~Param();

	// save in SVES format
	void Save (const char *filename);

	int Load (const char *filename, int loadType, unsigned int dwnsmp);
	int Load (const char *filename, unsigned int dwnsmp = 1); // load SVES
	int LoadRaw (const char *filename, unsigned int dwnsmp = 1);
	int LoadHTK (const char *filename, unsigned int dwnsmp = 1); 
	
	// load on feature vectors specified in 'frameIdxs', which is a list: each element represents a tuple (idx[0], idx[1])
	// feature vectors with indexes from interval [ idx[0], idx[1] ) are accumulated 
	int Load (const char *filename, const std::list < std::vector<unsigned int> >& frameIdxs, int loadType); 
	int Load (const char *filename, const std::list < std::vector<unsigned int> >& frameIdxs); // load SVES
	int LoadRaw (const char *filename, const std::list < std::vector<unsigned int> >& frameIdxs); 	
	int LoadHTK (const char *filename, const std::list < std::vector<unsigned int> >& frameIdxs); 	

	unsigned int GetVectorDim() const;
	unsigned int GetNumberOfVectors() const;
	unsigned int GetSamplePeriod() const;
	void SetSamplePeriod(unsigned int period);

	bool existSmplPeriod() const;
	float*** GetVectors();
	float* GetCFs();

	// feature vectors from 'data' are copied to this class; 
	// . if 'certaintyFactor = NULL' => it is assumed that CF is saved as the last dimension in 'data',
	//	thus it is assumed that	the true dimension of feature vectors is (dim - 1) !!!	
	template <typename T>
	void ExportInternal (T** data, unsigned int nsamples, unsigned int dim, 
		T* certaintyFactor = NULL);


	// if linkPointer = true => only pointer to vectors is stored, otherwise the vectors are copied;
	// if same feature vectors are added twice, only one can be added with linkPointer=true, otherwise
	// a runtime error will occur when the destructor of this class is called, since the same block of
	// memory will be deleted twice
	void Add (const Param& prm, bool linkPointer = false);

	void CleanMemory (bool release_vecs = false);

	friend class CParamKW;

protected:
	float **Vectors;		// feature vectors
	float *CF;				// certainity factor for each feature vector
	unsigned int Dim;				// dimension of feature vectors
	unsigned int NumberOfVectors;	// number of feature vectors
	unsigned int samplePeriod;		// sample period: available in HTK file format, but not in SVES
	bool existsSamplePeriod;
	std::list <unsigned int> nvectorsInBlock;
	
	// export data to one memory block
	unsigned int WriteBlock (char* StartAdr) const;

	// size of memory (in Bytes) needed to store the content of this object
	unsigned int GetBlockSize() const;

	unsigned int ReadBlock (const char* StartAdr);

	// forbidden
	Param (const Param& prm);
	Param& operator= (const Param& prm);
};



inline 
bool Param::existSmplPeriod() const {
	return existsSamplePeriod;
}



inline 
unsigned int Param::GetSamplePeriod() const {
	return samplePeriod;
}



inline 
void Param::SetSamplePeriod(unsigned int period) {
	samplePeriod = period;
	existsSamplePeriod = true;
}



inline 
unsigned int Param::GetVectorDim() const {
	return Dim;
}



inline 
unsigned int Param::GetNumberOfVectors() const {
	return NumberOfVectors;
}



inline 
float*** Param::GetVectors() {
	return &Vectors;
}



inline 
float* Param::GetCFs() {
	return CF;
}



template <typename T>
void Param::ExportInternal (T** data, unsigned int nsamples, unsigned int dim, 
							T* certaintyFactor) 
{	
	if(nsamples < 1 || data == NULL)
		return;

	unsigned int D = dim - 1; // CF save in the last dimension of data
	if (certaintyFactor != NULL)
		D = dim; // CF saved separately in the array 'certaintyFactor'
	
	if (NumberOfVectors > 0 && D != Dim)
			throw std::runtime_error("Param::ExportInternal(): dimension mismatch!");	

	float **vecBuff = NULL, *cfBuff = NULL;
	try {
		vecBuff = new float* [NumberOfVectors + nsamples];		
		vecBuff[NumberOfVectors] = new float [nsamples * D];
		for(unsigned int i = 0; i <  nsamples; i++) 
			vecBuff[NumberOfVectors + i] = &vecBuff[NumberOfVectors][i * D];

		cfBuff = new float [NumberOfVectors + nsamples];
	}
	catch (std::bad_alloc&) {
		delete [] vecBuff;
		delete [] cfBuff;
		throw std::runtime_error("Param::ExportInternal(): Memory alocation error.");
	}

	
	// copy previous data
	if(NumberOfVectors > 0) {
		memcpy(vecBuff, Vectors, sizeof(float*) * NumberOfVectors);		
		memcpy(cfBuff, CF, sizeof(float) * NumberOfVectors);
				
		delete [] Vectors;
		delete [] CF;
	}

	for(unsigned int i = 0; i < nsamples; i++) {
		if(certaintyFactor == NULL)
			cfBuff[NumberOfVectors + i] = static_cast <float> (data[i][D]);
		else
			cfBuff[NumberOfVectors + i] = static_cast <float> (certaintyFactor[i]);

		for(unsigned int j = 0; j < D; j++)
			vecBuff[NumberOfVectors + i][j] = static_cast <float> (data[i][j]);
	}

	Vectors = vecBuff;
	CF = cfBuff;

	NumberOfVectors += nsamples;
	nvectorsInBlock.push_back(nsamples);

	Dim = D;
}

#endif

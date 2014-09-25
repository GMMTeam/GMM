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

#include "param/OL_Param.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>

using namespace std;


Param::Param() 
: Vectors (NULL),
CF (NULL),
Dim (0),
NumberOfVectors (0),
samplePeriod (0),
existsSamplePeriod (false)
{} 



Param::~Param() {
	CleanMemory();
} //destruktor



void Param::CleanMemory(bool release_vecs) {
	
	if (NumberOfVectors > 0)
	{
		if(!release_vecs) 
		{
			unsigned int shift = 0;
			while (!nvectorsInBlock.empty()) {
				delete [] Vectors[shift];
				shift += nvectorsInBlock.front();
				nvectorsInBlock.pop_front();					
			}

		}				
	}		
	delete [] Vectors;
	delete [] CF;

	Dim = 0;
	NumberOfVectors = 0;
	nvectorsInBlock.clear();
	//samplePeriod = 0;
	//existsSamplePeriod = false;
	Vectors = NULL;
	CF = NULL;
} 



unsigned int Param::GetBlockSize() const {
	return (Dim + 1) * NumberOfVectors * sizeof(float) + 2 * sizeof(__int32);
} 



unsigned int Param::WriteBlock (char* StartAdr) const {

	char* p = StartAdr;

	//ulozeni dimenze
	memcpy(p, &Dim, sizeof(__int32));
	p += sizeof(__int32);

	//ulozeni poctu vektoru
	memcpy(p, &NumberOfVectors, sizeof(__int32));
	p += sizeof(__int32);

	if (NumberOfVectors < 1 || Dim < 1)
		return p - StartAdr;

	//ulozeni CF
	memcpy(p, CF, sizeof(float) * NumberOfVectors);
	p += sizeof(float) * NumberOfVectors;

	//ulozeni jednotlivych vektoru
	unsigned int shift = 0;	
	list<unsigned int>::const_iterator it;	
	for (it = nvectorsInBlock.begin(); it != nvectorsInBlock.end(); ++it) {
		unsigned int nsamples = *it;
		
		memcpy(p, Vectors[shift], sizeof(float) * Dim * nsamples);
		p += sizeof(float) * Dim * nsamples;
		
		shift += nsamples;
	}

	return p - StartAdr;

} 



unsigned int Param::ReadBlock(const char* StartAdr) {
	
	__int32 foo;

	// uvolneni pripadne stavajici pameti
	CleanMemory();
	
	const char *p = StartAdr;	

    // nacteni dimenze	
	memcpy(&foo, p, sizeof(__int32));
	Dim = static_cast <unsigned int> (foo);	
	p += sizeof(__int32);

	// ulozeni poctu vektoru
	memcpy(&foo, p, sizeof(__int32));
	NumberOfVectors = static_cast <unsigned int> (foo);	
	p += sizeof(__int32);

	// ked nemame ziadne vektory, ukoncime
	if(NumberOfVectors < 1 || Dim < 1)
		return p - StartAdr;

	nvectorsInBlock.push_back(NumberOfVectors);

	CF = new float [NumberOfVectors];
	Vectors = new float* [NumberOfVectors]; 
	Vectors[0] = new float [Dim * NumberOfVectors];

	//nacteni CF
	memcpy(CF, p, sizeof(float) * NumberOfVectors);
	p += sizeof(float) * NumberOfVectors;

	//nacteni vektoru
	memcpy(Vectors[0], p, sizeof(float) * Dim * NumberOfVectors);
	p += sizeof(float) * Dim * NumberOfVectors;
	
	for(unsigned int i = 1; i < NumberOfVectors; i++) 
		Vectors[i] = &Vectors[0][i * Dim];

	return p - StartAdr;

} 



void Param::Save (const char* filename) 
{
	ofstream file;
	file.open(filename, ios::out | ios::trunc | ios::binary);

	if(file.fail()) {
		string error_report = std::string("Param::Save(): file ") + filename + " could not be opened for writing";
		throw runtime_error(error_report.c_str());
	}

	file.exceptions(ofstream::eofbit | ofstream::failbit | ofstream::badbit);

	try {
		//zapis ID-hlavicky
		file.write("SV-ES-PARAM", 11);

		//ulozeni dimenze
		__int32 __D = static_cast<__int32> (Dim);
		file.write(reinterpret_cast <char *> (&__D), sizeof(__int32));

		//ulozeni poctu vektoru
		__int32 __N = static_cast<__int32> (NumberOfVectors);
		file.write(reinterpret_cast <char *> (&__N), sizeof(__int32));

		if (NumberOfVectors > 0) {			
			file.write(reinterpret_cast <char *> (CF), sizeof(float) * NumberOfVectors);

			//ulozeni jednotlivych vektoru
			unsigned int shift = 0;
			list<unsigned int>::iterator it;	
			for (it = nvectorsInBlock.begin(); it != nvectorsInBlock.end(); ++it) {
				unsigned int nsamples = *it;
				file.write(reinterpret_cast <char *> (Vectors[shift]), sizeof(float) * Dim * nsamples);

				shift += nsamples;
			}
		}
	}
	catch (ofstream::failure& e) {
		cout << "Param::Save(): Exception writing file" << endl << e.what() << endl;
	}

	file.close();
}



int Param::Load (const char *filename, int loadType, unsigned int dwnsmp) {
	int iRet;
	switch (loadType)
	{
	case(SVES_PRM_IN): 
		iRet = Load (filename, dwnsmp);
		break;
	case(HTK_PRM_IN): 
		iRet = LoadHTK (filename, dwnsmp);
		break;
	case(RAW_PRM_IN): 
		iRet = LoadRaw (filename, dwnsmp);
		break;
	default:
		throw runtime_error("Param::Load(): Unknown input data type specified");
	}

	return iRet;
}



int Param::Load(const char *filename, unsigned int dwnsmp)
{
	// uvolneni pripadne stavajici pameti
	CleanMemory();
	
	ifstream file;
	file.open(filename, ios::binary);

	if(file.fail()) {
		return -1;
		//string error_report = std::string("Param::Load(): file ") + filename + " could not be opened for reading";
		//throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);

	// read file header
	char header[12];	
	header[11] = '\0';

	file.read(header, 11);		
	if (strcmp(header, "SV-ES-PARAM") != 0) {
		string error_report = std::string("Param::Load(): file's ") + filename + " format different from SV-ES-PARAM";
		throw runtime_error(error_report.c_str());
	}

	try {
		// read dimension
		__int32 __foo;
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		Dim = static_cast<unsigned int> (__foo);

		// read number of samples
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		unsigned int nsamples = static_cast<unsigned int> (__foo);

		// set number of samples after downsampling,
		// dnwsmp = 1 means no downsampling! dwnsmp = 2 means: take each second frame (skip 1 frame)
		if (dwnsmp > 1)
			NumberOfVectors = nsamples / dwnsmp + (nsamples % dwnsmp > 0);			
		else			
			NumberOfVectors = nsamples;
		
		if (NumberOfVectors > 0 && Dim > 0)
		{
			nvectorsInBlock.push_back(NumberOfVectors);

			// alloc
			CF = new float [NumberOfVectors];
			Vectors = new float* [NumberOfVectors]; 
			Vectors[0] = NULL;
			Vectors[0] = new float [Dim * NumberOfVectors];

			// assign pointers
			for(unsigned int i = 1; i < NumberOfVectors; i++) 
				Vectors[i] = &Vectors[0][i * Dim];

			if (dwnsmp > 1) 
			{
				// load CFs
				for (unsigned int i = 0; i < NumberOfVectors - 1; i++) {
					file.read(reinterpret_cast <char *> (&CF[i]), sizeof(float));
					file.seekg(sizeof(float) * (dwnsmp-1), ios_base::cur);
				}
				file.read(reinterpret_cast <char *> (&CF[NumberOfVectors - 1]), sizeof(float));
				
				// skip rest of CFs
				unsigned int skip = nsamples - (NumberOfVectors - 1) * dwnsmp - 1;
				file.seekg(sizeof(float) * skip, ios_base::cur);

				// load vectors
				for (unsigned int i = 0; i < NumberOfVectors - 1; i++) {
					file.read(reinterpret_cast <char *> (Vectors[i]), sizeof(float) * Dim);
					file.seekg(sizeof(float) * (dwnsmp-1) * Dim, ios_base::cur);
				}
				file.read(reinterpret_cast <char *> (Vectors[NumberOfVectors - 1]), sizeof(float) * Dim);
			}
			else {
				file.read(reinterpret_cast <char *> (CF), sizeof(float) * NumberOfVectors);
				file.read(reinterpret_cast <char *> (Vectors[0]), sizeof(float) * Dim * NumberOfVectors);				
			}
		}
		else {
			NumberOfVectors = 0;
			Dim = 0;
		}
	}
	catch (bad_alloc&) {
		file.close();
		ostringstream error_report;
		error_report << "Param::LoadHTK(): Not enough memory when reading file " << filename << " [NSamples = " << NumberOfVectors << ", Dim = " << Dim << "]\n(bad file format (HTK | SVES)?)";

		if (Vectors != NULL)
			delete [] Vectors[0];
		NumberOfVectors = 0;

		throw runtime_error(error_report.str().c_str());
	}

	catch (ifstream::failure&) {		
		file.close();
		
		string error_report = std::string("Param::Load(): Error occured when reading file ") + filename;
		throw runtime_error(error_report.c_str());
	}

	file.close();
	return 0;
}



int Param::LoadRaw(const char *filename, unsigned int dwnsmp)
{
	// uvolneni pripadne stavajici pameti
	CleanMemory();
	
	ifstream file;
	file.open(filename, ios::binary);

	if(file.fail()) {
		return -1;
		//string error_report = std::string("Param::Load(): file ") + filename + " could not be opened for reading";
		//throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);

	try {
		// read number of samples
		__int32 __foo;
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		unsigned int nsamples = static_cast<unsigned int> (__foo);
		
		// read dimension
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		Dim = static_cast<unsigned int> (__foo);		

		if (nsamples < 1) {
			file.seekg(0, file.end);
			unsigned int fsize = file.tellg();
			fsize -= 2 * sizeof(__int32);
			nsamples = fsize / Dim / sizeof(float);
			file.seekg(2*sizeof(__int32), file.beg);
		}

		// set number of samples after downsampling,
		// dnwsmp = 1 means no downsampling! dwnsmp = 2 means: take each second frame (skip 1 frame)
		if (dwnsmp > 1)
			NumberOfVectors = nsamples / dwnsmp + (nsamples % dwnsmp > 0);			
		else			
			NumberOfVectors = nsamples;
		
		if (NumberOfVectors > 0 && Dim > 0)
		{
			nvectorsInBlock.push_back(NumberOfVectors);

			// alloc
			CF = new float [NumberOfVectors];
			for (unsigned int ncf = 0; ncf < NumberOfVectors; ncf++)
				CF[ncf] = 1.0f;
			Vectors = new float* [NumberOfVectors]; 
			Vectors[0] = NULL;
			Vectors[0] = new float [Dim * NumberOfVectors];

			// assign pointers
			for(unsigned int i = 1; i < NumberOfVectors; i++) 
				Vectors[i] = &Vectors[0][i * Dim];

			if (dwnsmp > 1) 
			{
				// load vectors
				for (unsigned int i = 0; i < NumberOfVectors - 1; i++) {
					file.read(reinterpret_cast <char *> (Vectors[i]), sizeof(float) * Dim);
					file.seekg(sizeof(float) * (dwnsmp-1) * Dim, ios_base::cur);
				}
				file.read(reinterpret_cast <char *> (Vectors[NumberOfVectors - 1]), sizeof(float) * Dim);
			}
			else
				file.read(reinterpret_cast <char *> (Vectors[0]), sizeof(float) * Dim * NumberOfVectors);				
		}
		else {
			NumberOfVectors = 0;
			Dim = 0;
		}
	}
	catch (bad_alloc&) {
		file.close();
		ostringstream error_report;
		error_report << "Param::LoadHTK(): Not enough memory when reading file " << filename << " [NSamples = " << NumberOfVectors << ", Dim = " << Dim << "]\n(bad file format (HTK | SVES)?)";

		if (Vectors != NULL)
			delete [] Vectors[0];
		NumberOfVectors = 0;

		throw runtime_error(error_report.str().c_str());
	}

	catch (ifstream::failure&) {		
		file.close();
		
		string error_report = std::string("Param::Load(): Error occured when reading file ") + filename;
		throw runtime_error(error_report.c_str());
	}

	file.close();
	return 0;
}



namespace {
	void LoadVectorHTK (ifstream& file, float *vector, unsigned int dim) {

		char* d;
		float numero; 

		for(unsigned int i = 0; i < dim; i++) 
		{
			d = reinterpret_cast <char *> (&numero);
			file.read(&d[3], sizeof(char));
			file.read(&d[2], sizeof(char));
			file.read(&d[1], sizeof(char));
			file.read(&d[0], sizeof(char));

			vector[i] = numero;
		}
	}
}


int Param::LoadHTK (const char *filename, unsigned int dwnsmp) 
{
	CleanMemory();
	
	ifstream file;
	file.open(filename, ios::binary);

	if(file.fail()) {
		return -1;
		//string error_report = std::string("Param::LoadHTK(): file ") + filename + " could not be opened for reading";
		//throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);

	try {
		unsigned int delka, perioda;
		short pbytu, kind;  
		char *d;

		d = reinterpret_cast <char *> (&delka);
		file.read(&d[3], sizeof(char));
		file.read(&d[2], sizeof(char));
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		d = reinterpret_cast <char *> (&perioda);
		file.read(&d[3], sizeof(char));
		file.read(&d[2], sizeof(char));
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		d = reinterpret_cast <char *> (&pbytu);
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		d = reinterpret_cast <char *> (&kind);
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		Dim = static_cast <unsigned int> (pbytu/4);
		unsigned int nsamples = static_cast <unsigned int> (delka);

		if (dwnsmp > 1)
			NumberOfVectors = nsamples / dwnsmp + (nsamples % dwnsmp > 0);
		else {
			NumberOfVectors = nsamples;
			dwnsmp = 1; // would cause problems if dwnsmp = 0
		}
		
		if(NumberOfVectors > 0 && Dim > 0) {
			
			nvectorsInBlock.push_back(NumberOfVectors);

			CF = new float [NumberOfVectors];
			Vectors = new float* [NumberOfVectors]; 
			Vectors[0] = NULL;
			Vectors[0] = new float [Dim * NumberOfVectors];

			// assign pointers & fill CF & load vectors
			for(unsigned int j = 0; j < NumberOfVectors - 1; j++) 
			{	
				CF[j] = 1.0f;
				Vectors[j] = &Vectors[0][j * Dim];
				LoadVectorHTK (file, Vectors[j], Dim);

				file.seekg(sizeof(float) * (dwnsmp-1) * Dim, ios_base::cur);
			}  
			CF[NumberOfVectors - 1] = 1.0f;
			Vectors[NumberOfVectors - 1] = &Vectors[0][(NumberOfVectors - 1) * Dim];
			LoadVectorHTK(file, Vectors[NumberOfVectors - 1], Dim);

			samplePeriod = perioda;
			existsSamplePeriod = true;
		}
		else {
			NumberOfVectors = 0;
			Dim = 0;
		}
	}
	catch (bad_alloc&) {
		file.close();
		ostringstream error_report;
		error_report << "Param::LoadHTK(): Not enough memory when reading file " << filename << " [NSamples = " << NumberOfVectors << ", Dim = " << Dim << "]\n (bad file format (HTK | SVES)?)";

		if (Vectors != NULL)
			delete [] Vectors[0];
		NumberOfVectors = 0;
		
		throw runtime_error(error_report.str().c_str());
	}
	catch (ifstream::failure&) {		
		file.close();
		
		string error_report = std::string("Param::LoadHTK(): Error occured when reading file ") + filename;
		throw runtime_error(error_report.c_str());
	}

	file.close();
	return 0;
}



int Param::Load (const char *filename, const std::list < std::vector<unsigned int> >& frameIdxs, int loadType) {
	int iRet;
	switch (loadType)
	{
	case(SVES_PRM_IN): 
		iRet = Load (filename, frameIdxs);
		break;
	case(HTK_PRM_IN): 
		iRet = LoadHTK (filename, frameIdxs);
		break;
	case(RAW_PRM_IN): 
		iRet = LoadRaw (filename, frameIdxs);
		break;
	default:
		throw runtime_error("Param::Load(): Unknown input data type specified");
	}

	return iRet;
}



int Param::Load (const char *filename, const list < vector<unsigned int> >& frameIdxs) 
{
	assert (existsSamplePeriod);
		//throw runtime_error("Param::Load(): set sample period first! (not present in SVES files)");

	// uvolneni pripadne stavajici pameti
	CleanMemory();
	
	ifstream file;
	file.open(filename, ios::binary);

	if(file.fail()) {
		return -1;
		//string error_report = std::string("Param::Load(): file ") + filename + " could not be opened for reading";
		//throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);

	// read file header
	char header[12];	
	header[11] = '\0';

	file.read(header, 11);		
	if (strcmp(header, "SV-ES-PARAM") != 0) {
		string error_report = std::string("Param::Load(): files' ") + filename + " format different from SV-ES-PARAM";
		throw runtime_error(error_report.c_str());
	}

	try {
		// read dimension
		__int32 __foo;
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		Dim = static_cast<unsigned int> (__foo);

		// read number of samples
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		unsigned int nsamples = static_cast<unsigned int> (__foo);
		
		unsigned int fbegin, fend, maxfend = 0;
		list < vector<unsigned int> >::const_iterator it;
		for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
			fbegin = (*it).at(0) / samplePeriod;
			fend = (*it).at(1) / samplePeriod;
			NumberOfVectors += fend - fbegin;
			
			if (fbegin >= fend || fbegin >= nsamples) {
				file.close();

				string error_report = std::string("Param::Load(): In file ") + filename + ": prm segment_begin >= (segment_end OR number_of_samples)";
				throw runtime_error(error_report.c_str());
			}

			if (fend > maxfend)
				maxfend = fend;

			if (fend > nsamples && fbegin < nsamples)
				NumberOfVectors += nsamples - fbegin;
			else
				NumberOfVectors += fend - fbegin;
		}

		if (maxfend > nsamples) {
			cerr << "WARNING: Param::Load(): htk file " << filename << " too short: " << maxfend - nsamples << " frames missing" << endl;
			//file.close();

			//string error_report = std::string("Param::Load(): File ") + filename + " does not contain desired amount of frames";
			//throw runtime_error(error_report.c_str());
		}

		if (NumberOfVectors > 0 && Dim > 0)
		{
			nvectorsInBlock.push_back(NumberOfVectors);

			// alloc
			CF = new float [NumberOfVectors];
			Vectors = new float* [NumberOfVectors]; 
			Vectors[0] = NULL;
			Vectors[0] = new float [Dim * NumberOfVectors];

			// read CFs & prepare Vectors
			unsigned int counter = 0, last_idx = 0, Nskip;
			for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
				fbegin = (*it).at(0) / samplePeriod;
				fend = (*it).at(1) / samplePeriod;
				
				if (fbegin < last_idx) {
					file.close();
					throw runtime_error("Param::Load(): Incorrect frame intervals (not ascending!) specified");
				}
				if (fend > nsamples)
					fend = nsamples;

				Nskip = (fbegin - last_idx);
				last_idx = fend;

				file.seekg(sizeof(float) * Nskip, ios_base::cur);
				file.read(reinterpret_cast <char *> (&CF[counter]), sizeof(float) * (fend - fbegin));				
				for(unsigned int j = 0; j < fend - fbegin; j++) {					
					Vectors[counter] = &Vectors[0][counter * Dim];
					counter++;
				}
			}

			assert (counter == NumberOfVectors);

			// read Vectors
			counter = last_idx = 0;
			for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
				fbegin = (*it).at(0) / samplePeriod;
				fend = (*it).at(1) / samplePeriod;				
				Nskip = (fbegin - last_idx);
				last_idx = fend;

				file.seekg(sizeof(float) * Nskip * Dim, ios_base::cur);				
				file.read(reinterpret_cast <char *> (Vectors[counter]), sizeof(float) * (fend - fbegin) * Dim);
				counter += fend - fbegin;
			}

			assert (counter == NumberOfVectors);
		}
		else {
			NumberOfVectors = 0;
			Dim = 0;
		}
	}
	catch (bad_alloc&) {
		file.close();
		ostringstream error_report;
		error_report << "Param::Load(): Not enough memory when reading file " << filename << " [NSamples = " << NumberOfVectors << ", Dim = " << Dim << "]\n(bad file format (HTK | SVES)?)";

		if (Vectors != NULL)
			delete [] Vectors[0];
		NumberOfVectors = 0;

		throw runtime_error(error_report.str().c_str());
	}

	catch (ifstream::failure&) {		
		file.close();
		
		string error_report = std::string("Param::Load(): Error occured when reading file ") + filename;
		throw runtime_error(error_report.c_str());
	}

	file.close();
	return 0;
}



int Param::LoadRaw (const char *filename, const list < vector<unsigned int> >& frameIdxs) 
{
	assert (existsSamplePeriod);
		//throw runtime_error("Param::Load(): set sample period first! (not present in SVES files)");

	// uvolneni pripadne stavajici pameti
	CleanMemory();
	
	ifstream file;
	file.open(filename, ios::binary);

	if(file.fail()) {
		return -1;
		//string error_report = std::string("Param::Load(): file ") + filename + " could not be opened for reading";
		//throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);

	try {
		// read dimension
		__int32 __foo;
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		unsigned int nsamples = static_cast<unsigned int> (__foo);

		// read number of samples
		file.read(reinterpret_cast <char *> (&__foo), sizeof(__int32));
		Dim = static_cast<unsigned int> (__foo);		
		
		unsigned int fbegin, fend, maxfend = 0;
		list < vector<unsigned int> >::const_iterator it;
		for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
			fbegin = (*it).at(0) / samplePeriod;
			fend = (*it).at(1) / samplePeriod;
			NumberOfVectors += fend - fbegin;
			
			if (fbegin >= fend || fbegin >= nsamples) {
				file.close();

				string error_report = std::string("Param::Load(): In file ") + filename + ": prm segment_begin >= (segment_end OR number_of_samples)";
				throw runtime_error(error_report.c_str());
			}

			if (fend > maxfend)
				maxfend = fend;

			if (fend > nsamples && fbegin < nsamples)
				NumberOfVectors += nsamples - fbegin;
			else
				NumberOfVectors += fend - fbegin;
		}

		if (maxfend > nsamples) {
			cerr << "WARNING: Param::LoadHTK(): htk file " << filename << " too short: " << maxfend - nsamples << " frames missing" << endl;
			//file.close();

			//string error_report = std::string("Param::Load(): File ") + filename + " does not contain desired amount of frames";
			//throw runtime_error(error_report.c_str());
		}

		if (NumberOfVectors > 0 && Dim > 0)
		{
			nvectorsInBlock.push_back(NumberOfVectors);

			// alloc
			CF = new float [NumberOfVectors];
			for (unsigned int ncf = 0; ncf < NumberOfVectors; ncf++)
				CF[ncf] = 1.0f;

			Vectors = new float* [NumberOfVectors]; 
			Vectors[0] = NULL;
			Vectors[0] = new float [Dim * NumberOfVectors];

			// read & prepare Vectors
			unsigned int counter = 0, last_idx = 0, Nskip;
			for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
				fbegin = (*it).at(0) / samplePeriod;
				fend = (*it).at(1) / samplePeriod;
				
				if (fbegin < last_idx) {
					file.close();
					throw runtime_error("Param::Load(): Incorrect frame intervals (not ascending!) specified");
				}
				if (fend > nsamples)
					fend = nsamples;

				Nskip = (fbegin - last_idx);
				last_idx = fend;

				file.seekg(sizeof(float) * Nskip, ios_base::cur);				
				for(unsigned int j = 0; j < fend - fbegin; j++) {					
					Vectors[counter] = &Vectors[0][counter * Dim];
					counter++;
				}
			}

			assert (counter == NumberOfVectors);

			// read Vectors
			counter = last_idx = 0;
			for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
				fbegin = (*it).at(0) / samplePeriod;
				fend = (*it).at(1) / samplePeriod;				
				Nskip = (fbegin - last_idx);
				last_idx = fend;

				file.seekg(sizeof(float) * Nskip * Dim, ios_base::cur);				
				file.read(reinterpret_cast <char *> (Vectors[counter]), sizeof(float) * (fend - fbegin) * Dim);
				counter += fend - fbegin;
			}

			assert (counter == NumberOfVectors);
		}
		else {
			NumberOfVectors = 0;
			Dim = 0;
		}
	}
	catch (bad_alloc&) {
		file.close();
		ostringstream error_report;
		error_report << "Param::Load(): Not enough memory when reading file " << filename << " [NSamples = " << NumberOfVectors << ", Dim = " << Dim << "]\n(bad file format (HTK | SVES)?)";

		if (Vectors != NULL)
			delete [] Vectors[0];
		NumberOfVectors = 0;

		throw runtime_error(error_report.str().c_str());
	}

	catch (ifstream::failure&) {		
		file.close();
		
		string error_report = std::string("Param::Load(): Error occured when reading file ") + filename;
		throw runtime_error(error_report.c_str());
	}

	file.close();
	return 0;
}



int Param::LoadHTK (const char *filename, const list < vector<unsigned int> >& frameIdxs)
{
	CleanMemory();
	
	ifstream file;
	file.open(filename, ios::binary);

	if(file.fail()) {
		return -1;
		//string error_report = std::string("Param::LoadHTK(): file ") + filename + " could not be opened for reading";
		//throw runtime_error(error_report.c_str());
	}

	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);

	try {
		unsigned int delka, perioda;
		short pbytu, kind;  
		char *d;

		d = reinterpret_cast <char *> (&delka);
		file.read(&d[3], sizeof(char));
		file.read(&d[2], sizeof(char));
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		d = reinterpret_cast <char *> (&perioda);
		file.read(&d[3], sizeof(char));
		file.read(&d[2], sizeof(char));
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		d = reinterpret_cast <char *> (&pbytu);
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		d = reinterpret_cast <char *> (&kind);
		file.read(&d[1], sizeof(char));
		file.read(&d[0], sizeof(char));

		Dim = static_cast <unsigned int> (pbytu/4);
		unsigned int nsamples = static_cast <unsigned int> (delka);

		unsigned int fbegin, fend, maxfend = 0;
		list < vector<unsigned int> >::const_iterator it;
		for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
			fbegin = (*it).at(0) / perioda;
			fend = (*it).at(1) / perioda;			
			
			if (fbegin >= fend || fbegin >= nsamples) {
				file.close();

				string error_report = std::string("Param::LoadHTK(): In file ") + filename + ": prm segment_begin >= (segment_end OR number_of_samples)";
				throw runtime_error(error_report.c_str());
			}

			if (fend > maxfend)
				maxfend = fend;

			if (fend > nsamples && fbegin < nsamples)
				NumberOfVectors += nsamples - fbegin;
			else
				NumberOfVectors += fend - fbegin;
		}

		if (maxfend > nsamples) {
			cerr << "WARNING: Param::LoadHTK(): htk file " << filename << " too short: " << maxfend - nsamples << " frames missing" << endl;
			//file.close();

			//string error_report = std::string("Param::LoadHTK(): File ") + filename + " does not contain desired amount of frames";
			//throw runtime_error(error_report.c_str());
		}
	
		if(NumberOfVectors > 0 && Dim > 0) {
			
			nvectorsInBlock.push_back(NumberOfVectors);

			CF = new float [NumberOfVectors];
			Vectors = new float* [NumberOfVectors]; 
			Vectors[0] = NULL;
			Vectors[0] = new float [Dim * NumberOfVectors];
			
			unsigned int counter = 0, last_idx = 0, Nskip;
			for (it = frameIdxs.begin(); it != frameIdxs.end(); it++) {
				fbegin = (*it).at(0) / perioda;
				fend = (*it).at(1) / perioda;
				
				if (fbegin < last_idx) {
					file.close();
					throw runtime_error("Param::LoadHTK(): Incorrect frame intervals (not ascending!) specified");
				}
				if (fend > nsamples)
					fend = nsamples;

				Nskip = (fbegin - last_idx) * Dim;
				last_idx = fend;

				file.seekg(sizeof(float) * Nskip, ios_base::cur);
				for(unsigned int j = 0; j < fend - fbegin; j++) {		
					CF[counter] = 1.0f;
					Vectors[counter] = &Vectors[0][counter * Dim];

					LoadVectorHTK (file, Vectors[counter], Dim);
					counter++;
				}
			}

			assert (counter == NumberOfVectors);

			samplePeriod = perioda;
			existsSamplePeriod = true;
		}
		else {
			NumberOfVectors = 0;
			Dim = 0;
		}
	}
	catch (bad_alloc&) {
		file.close();		
		ostringstream error_report;
		error_report << "Param::LoadHTK(): Not enough memory when reading file " << filename << " [NSamples = " << NumberOfVectors << ", Dim = " << Dim << "]\n (bad file format (HTK | SVES)?)";

		if (Vectors != NULL)
			delete [] Vectors[0];
		NumberOfVectors = 0;

		throw runtime_error(error_report.str().c_str());
	}

	catch (ifstream::failure&) {		
		file.close();
		
		string error_report = std::string("Param::LoadHTK(): Error occured when reading file ") + filename;
		throw runtime_error(error_report.c_str());
	}

	file.close();
	return 0;
}



void Param::Add (const Param& prm, bool linkPointer)
{
	if(prm.NumberOfVectors < 1)
		return;

	if (NumberOfVectors > 0 && prm.Dim != Dim)
			throw std::runtime_error("Param::Add(): dimension mismatch!");	

	float **vecBuff = NULL, *cfBuff = NULL;
	try {
		vecBuff = new float* [NumberOfVectors + prm.NumberOfVectors];
		cfBuff = new float [NumberOfVectors + prm.NumberOfVectors];
		if (!linkPointer) {
			vecBuff[NumberOfVectors] = new float [prm.NumberOfVectors * prm.Dim];
			for(unsigned int i = 0; i < prm.NumberOfVectors; i++) 
				vecBuff[NumberOfVectors + i] = &vecBuff[NumberOfVectors][i * prm.Dim];		
		}
	}
	catch (std::bad_alloc&) {
		delete [] vecBuff;
		delete [] cfBuff;
		throw std::runtime_error("Param::ExportInternal(): Memory alocation error.");
	}

	
	//zkopirujeme pripadna predchozi data
	if(NumberOfVectors > 0) {
		memcpy(vecBuff, Vectors, sizeof(float*) * NumberOfVectors);		
		memcpy(cfBuff, CF, sizeof(float) * NumberOfVectors);
				
		delete [] Vectors;
		delete [] CF;
	}
	// zkopirujeme nove CF
	memcpy(cfBuff + NumberOfVectors, prm.CF, sizeof(float) * prm.NumberOfVectors);

	if (!linkPointer) {				
		unsigned int shift = 0;
		list <unsigned int>::const_iterator it;
		for (it = prm.nvectorsInBlock.begin(); it != prm.nvectorsInBlock.end(); it++) {
			unsigned int nsamples = *it;
			memcpy(vecBuff[NumberOfVectors] + shift, prm.Vectors[shift], sizeof(float) * prm.Dim * nsamples);
			shift += nsamples;
		}
		nvectorsInBlock.push_back(prm.NumberOfVectors);
	}
	else {
		list <unsigned int>::const_iterator it;
		for (it = prm.nvectorsInBlock.begin(); it != prm.nvectorsInBlock.end(); it++) 
			nvectorsInBlock.push_back(*it);
		memcpy(vecBuff + NumberOfVectors, prm.Vectors, sizeof(float*) * prm.NumberOfVectors);
	}

	Vectors = vecBuff;
	CF = cfBuff;

	NumberOfVectors += prm.NumberOfVectors;
	Dim = prm.Dim;
}


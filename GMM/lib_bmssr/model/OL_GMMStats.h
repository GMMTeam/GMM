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

#ifndef _OL_GMM_STATS_H_
#define _OL_GMM_STATS_H_

#include <vector>
#include <iostream>

template <typename TS, typename TV, typename TLL> class GMMStatsEstimator;
template <typename TS> class GMMStatsEstimator_CUDA;
class CModelAdapt; // REPLACE BY THE USE OF OPERATOR []
class Centralizer; // REPLACE BY THE USE OF OPERATOR []


template <typename T> 
class GMMStats {
public:
	struct mixStats {
		T mixProb;  // Mixture probability.
		T* mean;    // Divide at the end by mixProb
		            // to get a proper first moment.
		T* var;     // Divide at the end by mixProb 
				    // to get a proper second central moment.
		T* aux;     // auxiliary statistics - acc abs diff
		T aux2;		// auxiliary number     - sum sqrt(g)
		T aux3;		// auxiliary number     - sum(g) 
	};
	
	GMMStats();
	GMMStats (unsigned int modeldim, unsigned int nummix,
			  bool allocMeans = true, bool allocVars = false, 
			  bool allocFullVars = false, bool allocAux = false);
	GMMStats (const GMMStats<T>& gmmStats);
	~GMMStats();
		
	GMMStats<T>& operator= (const GMMStats<T>& gmmStats);

	// operator+ is not comutative (!), hence (a+b) != (b+a);
	// only statistics allocated in the sums' left-value will be summed
	const GMMStats<T> operator+ (const GMMStats<T>& rgmmStats) const;	
	GMMStats<T>& operator+= (const GMMStats<T>& rgmmStats);	

	// dangerous operator! you can do whatever you want with a mixture component 
	// (change pointers, values) without any additional error handling - use with caution!
	mixStats& operator[] (unsigned int mixidx);	

	// alloc does not initialize any values -> use reset()
	void alloc (const GMMStats& gmmStats) ;
	void alloc (unsigned int modeldim, unsigned int nummix,
			    bool allocMeans = true, bool allocVars = false, 
			    bool allocFullVars = false, bool allocAux = false) ;

		
	void deleteStats();
	void reset(); // Set statistics back to zero.
	
	bool isAllocated() const;
	bool getAllocStatus (bool& allocMeans, bool& allocVars, 
						 bool& allocFullVars, bool& allocAux) const;
	bool hasFullCov() const;
	
	unsigned int getMixNum() const;
	unsigned int getDim() const;
	unsigned int getVarDim() const;
	unsigned int getTotAccSamples() const;
	T getTotLogLike() const;
		
	T getProb (unsigned int mixidx) const;
	const T* getMean (unsigned int mixidx) const;
	const T* getVar (unsigned int mixidx) const;
	const T* getAux (unsigned int mixidx) const;
	const T getAux2 (unsigned int mixidx) const;
	const T getAux3 (unsigned int mixidx) const;
	
	void getMixStats (mixStats& ms_dest, unsigned int mixidx, 
					  bool allocate = true) const;  

	void save (const char* filename, bool saveTxt, 
			   bool saveMProbs = true, bool saveMeans = true, 
			   bool saveVars = true, bool saveAux = true) const;

	void load (const char* filename, bool loadTxt, 
			   bool loadMeans = true, bool loadVars = true);
	

	template <typename TS, typename TV, typename TLL> friend class GMMStatsEstimator;
	template <typename TS> friend class GMMStatsEstimator_CUDA;
	friend class CModelAdapt; // REPLACE BY THE USE OF OPERATOR []
	friend class Centralizer; // REPLACE BY THE USE OF OPERATOR []

protected:
	void initializeVariables();	
	void clone (const GMMStats& gmmStats);
	void allocMixStats (mixStats& ms, bool allocMeans, unsigned int mdim,
					    bool allocVars, unsigned int vdim, 
						bool allocAux, unsigned int adim) const;
	void deleteStats (unsigned int begin, unsigned int end);

	void loadBin (std::ifstream& ifile, bool loadMeans = true, bool loadVars = true);
	void loadTxt (std::ifstream& ifile, bool loadMeans = true, bool loadVars = true);


	void saveTxt (std::ofstream& ofile, bool saveMProbs = true,
				  bool saveMeans = true, bool saveVars = true) const;
	void saveBin (std::ofstream& ofile, bool saveMProbs = true,
				  bool saveMeans = true, bool saveVars = true) const;
	void saveAux (const char* filename) const;


private:
	bool _allocated;
	bool _allM, _allV, _allFV, _allA; // allocMeans, allocVars, allocFullVars

	unsigned int _dim, _varsize;
	unsigned int _totalAccSamples;

	std::vector<mixStats> _ms;
	T _totLogLike;
};

#include "model/OL_GMMStats.cpp"

#endif

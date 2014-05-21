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

#ifndef _OL_GMModel_
#define _OL_GMModel_

//velikost buffer pro nacitani ze souboru
#ifndef GMM_FILEBUFFERSIZE
#define GMM_FILEBUFFERSIZE	16000
#endif

//ulozeni cisel ve tvaru %15e (pro 1) nebo %15.10f (pro 0) .. pro textovy zapis)
#define DBGMM_WRITE_TXT_EXP		1	



class GMMMixture {
private:
	__int16	Dim;	//dimenze vektoru
	__int32	DimVar;	//dimenze vektoru
	float mixOcc;
	float *Mean;	//vektor strednich hodnot
	float *Var;		//vektor varianci
	float *VarDiag;		//vektor diagonal

public:	
	GMMMixture(int Dimension, bool full_cov = 0);
	~GMMMixture();

	//nastaveni hodnot
	int SetMean(float *Vector);
	int SetVar(float *Vector);
	inline void setMixOcc(float newMixOcc) {mixOcc = newMixOcc;}
	void Reset();

	friend class GMModel;
	friend class CModelAdapt;
}; //GMMMixture


class GMModel {
protected:	
	__int16	Dim;			//dimenze vekoru
	__int32	DimVar;			//dimenze vekoru variacie
	__int16	NumMixes;	    //pocet slozek	
	__int16	NumMixesNZW;	//pocet slozeks nenulovou vahou
	float *Weights;			//vektor vah jednotlivych slozek
	float *NormCoef;		//vektor normalizacnich koeficientu
	GMMMixture **Mixes;  //vektor pointru na jednotlive slozky
	
	void CleanMemory(void); //uvolneni pameti
public:
	
	GMModel();	//bez alokace pameti pro pole
	GMModel(int Dimension, int NumberOfMixes, bool full_cov);	// alokaci pameti pro pole	
	GMModel(GMModel *Original); //copy-constructor
	virtual ~GMModel();

	void copy(GMModel *Original);
	void reset();

	//nastaveni hodnot
	int SetNormCoef(float *Vector);
	int SetMixParam(int Mixture, float *Mean, float *Var, float Weight);
	int SetMixWeight(int MixtureIndex, float Weight);	

	//precteni hodnot
	int GetNormCoef(float *Vector);
	int GetMixParam(int Mixture, float *Mean, float *Var, float *Weight);

	//inline funkce
	inline virtual bool GetFullCovStatus() {return Dim != DimVar;}
	inline virtual int GetNumberOfMixtures() {return(NumMixes);}
	inline virtual int GetNumberOfNZWMixtures() {return NumMixesNZW;} // najprv treba volat RearrangeMixtures(); -> inicializuje NumMixesNZW!!
	inline virtual int GetDimension() {return(Dim);}
	inline virtual float GetMixtureWeight(int MixtureIndex) {return(Weights[MixtureIndex]);}
	inline virtual float *GetMixtureWeights() {return(Weights);}
	inline virtual float *GetMixtureMean(int MixtureIndex) {return(Mixes[MixtureIndex]->Mean);}
	inline virtual float *GetMixtureVar(int MixtureIndex) {return(Mixes[MixtureIndex]->Var);}
	virtual float *GetMixtureVarDiag(int MixtureIndex);
	inline virtual float GetMixtureOccup(int MixtureIndex) {return(Mixes[MixtureIndex]->mixOcc);}
	inline GMMMixture *GetMixture(int MixtureIndex) {return(Mixes[MixtureIndex]);}

	int Load(const char *FileName, bool loadTxt);
	int Save(const char *FileName, bool saveTxt); 

	//ulozeni v textovem a v binarnim formatu
	int WriteTXT(const char *FileName);
	int WriteBIN(const char *FileName);

	//nacteni v textovem a v binarnim formatu
	int ReadTXT(const char *FileName);
	int ReadBIN(const char *FileName);

	//pridani slozky (nekontroluje se podminka suma(weight =1)
	void AddMixture(float *Mean, float *Var, float Weight);
	
	//preusporadat slozky - slozky s nulovou vahou nakonec;
	//vrati pocet nulovych slozek
	virtual unsigned int RearrangeMixtures();

	//funkce pro export dat v bloku
	unsigned int GetBlockSize(void);
	unsigned int WriteBlock(void *StartAdr);
	unsigned int ReadBlock(void *StartAdr);
	
	friend class CModelAdapt;
}; // GMModel

#endif

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

#include "model/OL_GMModel.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
 


GMMMixture::GMMMixture(int Dimension, bool full_cov){
	this->Dim = Dimension;
	if (full_cov == 1)
		this->DimVar = Dimension * Dimension;
	else
		this->DimVar = Dimension;

	this->mixOcc = -1.0;

	//alokace pameti - mean
	this->Mean = new float[Dim];
	//alokace pameti - var
	this->Var = new float[DimVar];
	this->VarDiag = NULL;

	Reset();
} //konstruktor



GMMMixture::~GMMMixture(){
	if (Mean != NULL) delete[] Mean;
	if (Var != NULL) delete[] Var;
	if (VarDiag != NULL) delete[] VarDiag;
} //destruktor



void GMMMixture::Reset() {

	memset(Mean, 0, Dim * sizeof(float));		

	if (DimVar != Dim) {
		memset(Var, 0, DimVar * sizeof(float));
		for (unsigned int d = 0; d < Dim; d++) 
			Var[d * Dim + d] = 1;		
	}
	else memset(Var, 1, DimVar * sizeof(float));
}



int GMMMixture::SetMean(float *Vector){
	int i;
	
	if (this->Mean == NULL){
		printf("GMMMixture::SetMean() : Memory was not allocated.\n");
		return(1);
	}

	memcpy(Mean, Vector, sizeof(float) * Dim);
	//for(i=0;i<Dim;i++){
	//	Mean[i] = Vector[i];
	//} //for i

	return(0);
} //SetMean



int GMMMixture::SetVar(float *Vector){

	memcpy(Var, Vector, sizeof(float) * DimVar);	
	if (VarDiag != NULL) {
		for (unsigned int i = 0; i < Dim; i++)
			VarDiag[i] = Var[i*Dim + i];
	}

	return(0);
} //SetNormCoef


// ----------------------------------------------------
// class GMModel --------------------------------------

GMModel::GMModel(int Dimension, int NumberOfMixes, bool full_cov){
	this->Dim = Dimension;
	if (full_cov == 1)
		this->DimVar = Dimension * Dimension;
	else
		this->DimVar = Dimension;

	this->NumMixes = NumberOfMixes;
	this->NumMixesNZW = 0;

	//alokace pameti pro pole vah smesi
	if ((this->Weights = (float*) malloc(sizeof(float)*NumMixes)) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}

	//alokace pameti pro pole normalizacnich koeficientu
	if ((this->NormCoef = (float*) malloc(sizeof(float)*Dim)) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}

	//alokace pameti pro pole smesi
	if ((this->Mixes = (GMMMixture**) malloc(sizeof(GMMMixture*)*NumMixes)) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}
	int i;
	for(i=0;i<NumMixes;i++){
		if ((Mixes[i] = new GMMMixture(Dim, full_cov)) == NULL){
			printf("Cannot allocate memory.\n");
			exit(1);
		}
	} //for i

	reset();
} //konstruktor - pro vytvoreni modelu (napr. budou natrenovany parametry


void GMModel::reset() {	
	for(int i = 0; i < NumMixes; i++){
		Weights[i] = 0.0f;
	}

	for(int i = 0; i < Dim; i++){
		NormCoef[i] = 1.0;
	}
}


GMModel::GMModel(){
	this->Dim = 0;
	this->DimVar = 0;
	this->Mixes = NULL;	
	this->NormCoef = NULL;
	this->NumMixes = 0;
	this->Weights = NULL;
} //konstruktor - pro nacteni ze souboru



GMModel::GMModel(GMModel *Original){
	this->Dim = 0;
	this->DimVar = 0;
	this->Mixes = NULL;	
	this->NormCoef = NULL;
	this->NumMixes = 0;
	this->Weights = NULL;

	copy(Original);
} //kopirovaci konstruktor



float *GMModel::GetMixtureVarDiag(int MixtureIndex) {
	if (!GetFullCovStatus())
		return Mixes[MixtureIndex]->Var;
	else {
		if (Mixes[MixtureIndex]->VarDiag == NULL) {
			Mixes[MixtureIndex]->VarDiag = new float[Dim];
			for (unsigned int i = 0; i < Dim; i++)
				Mixes[MixtureIndex]->VarDiag[i] = Mixes[MixtureIndex]->Var[i*Dim + i];
		}		
		return Mixes[MixtureIndex]->VarDiag;
	}
}

void GMModel::copy(GMModel *Original) 
{
	if (NumMixes != Original->NumMixes || Dim != Original->Dim || DimVar != Original->DimVar) {
		CleanMemory();

		//kopie skalaru
		this->Dim = Original->Dim;
		this->DimVar = Original->DimVar;
		this->NumMixes = Original->NumMixes;

		//deep copy - normalizacni koeficient
		if ((this->NormCoef = (float*) malloc(sizeof(float)*Dim)) == NULL){
			printf("Cannot allocate memory.\n");
			exit(1);
		}
		//deep copy - jednotlive slozky
		if ((this->Weights = (float*) malloc(sizeof(float)*NumMixes)) == NULL){
			printf("Cannot allocate memory.\n");
			exit(1);
		}
		if ((this->Mixes = (GMMMixture**) malloc(sizeof(GMMMixture*)*NumMixes)) == NULL){
			printf("Cannot allocate memory.\n");
			exit(1);
		}
		for(int i=0;i<NumMixes;i++)
			Mixes[i] = new GMMMixture(Dim, Original->GetFullCovStatus());
	}
	//nastaveni hodnot - normalizacni koeficient
	SetNormCoef(Original->NormCoef);

	//nastaveni hodnot - jednotlive slozky	
	for(int i=0;i<NumMixes;i++){		
		SetMixParam(i, 
			Original->GetMixtureMean(i), 
			Original->GetMixtureVar(i), 
			Original->GetMixtureWeight(i));
	} //for i
}



GMModel::~GMModel(){
	this->CleanMemory();
} //destruktor



void GMModel::CleanMemory(void){
	if (Weights != NULL) free((void*) Weights);
	if (NormCoef != NULL) free((void*) NormCoef);
	if (Mixes != NULL){
		int i;
		for(i=0;i<NumMixes;i++){
			if (Mixes[i] != NULL) delete Mixes[i];
		} //for i
		free((void*) Mixes);
	}
	Weights = NULL;
	NormCoef = NULL;
	Mixes = NULL;
} //CleanMemory



int GMModel::SetNormCoef(float *Vector){	
	
	if (Vector == NULL)
		return(0);

	if (this->NormCoef == NULL){
		printf("GMModel::SetNormCoef() : Memory was not allocated.\n");
		return(1);
	}

	memcpy(NormCoef, Vector, sizeof(float) * Dim);
	//for(int i=0;i<Dim;i++){
	//	NormCoef[i] = Vector[i];
	//} //for i
	return(0);
} //SetNormCoef



int GMModel::SetMixParam(int Mixture, float *Mean, float *Var, float Weight){
	if (Mixture >= NumMixes){
		printf("GMModel::SetMixParam() : Too big mixture index\n");
		return(1);
	}
	if (Weights == NULL){
		printf("GMModel::SetMixParam() : Memory was not allocated\n");
		return(1);
	}
	this->Weights[Mixture] = Weight;

	if (Mixes[Mixture]->SetMean(Mean) != 0) return(0);
	if (Mixes[Mixture]->SetVar(Var) != 0) return(0);
	return(0);
} //SetMixParam



int GMModel::SetMixWeight(int MixtureIndex, float Weight) {

	if ((MixtureIndex >=0) && (MixtureIndex < NumMixes))
		this->Weights[MixtureIndex] = Weight;
	else return(1);

	return(0);
}



int GMModel::GetNormCoef(float *Vector){
	int i;
	for(i=0;i<this->Dim;i++){
		Vector[i] = NormCoef[i];
	} //for i

	return(0);
} //GetNormCoef



int GMModel::GetMixParam(int Mixture, float *Mean, float *Var, float *Weight){
	if (Mixture >= NumMixes){
		printf("GMModel::SetMixParam() : Too big mixture index\n");
		return(1);
	}
	*Weight = this->Weights[Mixture];

	int i;
	for(i=0;i<Dim;i++) Mean[i] = Mixes[Mixture]->Mean[i];
	for(i=0;i<DimVar;i++) Var[i] = Mixes[Mixture]->Var[i];

	return(0);
} //GetMixParam



//int GMModel::WriteTXT(const char *FileName){
//	FILE *fw;
//	int i;
//	int j;
//
//	//otevreni souboru
//	if ((fw = fopen(FileName, "w")) == NULL){
//		printf("Cannot open output file '%s'.\n", FileName);
//		return(1);
//	} //fopen
//
//	//ulozeni dimenze
//	fprintf(fw, "%d\n", this->Dim);
//	//ulozeni normalizacniho vektoru
//	for(i=0;i<Dim;i++){
//		#if (DBGMM_WRITE_TXT_EXP == 1)
//			fprintf(fw, "%15e\n", this->NormCoef[i]);
//		#else
//			fprintf(fw, "%15.10f\n", this->NormCoef[i]);
//		#endif
//	} //for i
//
//	//ulozeni poctu slozek
//	fprintf(fw, "%d\n", this->NumMixes);
//	
//	//ulozeni jednotlivych smesi
//	for(j=0;j<this->NumMixes;j++){
//		//weight
//		fprintf(fw, "%15e", this->Weights[j]);
//		if(GetFullCovStatus())
//			fprintf(fw, " F");
//		if(this->GetMixtureOccup(j) != -1) 
//			fprintf(fw, " <MIX_OCCUP> %15e", this->Mixes[j]->mixOcc);
//		fprintf(fw, "\n");
//
//		fprintf(fw, "\n");
//		//mean
//		for(i=0;i<Dim;i++){
//			fprintf(fw, "%15e\n", this->Mixes[j]->Mean[i]);
//		}
//		fprintf(fw, "\n");
//		
//		//var
//		if(GetFullCovStatus()) {
//			for(i=0;i<Dim;i++){
//				fprintf(fw, "%15e", this->Mixes[j]->Var[i*Dim + i]);
//				for(unsigned int ii=0;ii<Dim;ii++){
//					fprintf(fw, " %15e", this->Mixes[j]->Var[i*Dim + ii]);
//				}	
//				fprintf(fw, "\n");
//			}
//		}
//		else {
//			for(i=0;i<Dim;i++){
//				fprintf(fw, "%15e\n", this->Mixes[j]->Var[i]);
//			}
//		}
//		fprintf(fw, "\n");
//	} //for j
//
//	if (fclose(fw) == EOF){
//		printf("Cannot close file '%s'.\n", FileName);
//	} //close
//
//	return(0);
//} //WriteTXT
//
//
//
//int GMModel::WriteBIN(const char *FileName){
//	FILE *fw;
//	int j;
//
//	//otevreni souboru
//	if ((fw = fopen(FileName, "wb")) == NULL){
//		printf("Cannot open output file '%s'.\n", FileName);
//		return(1);
//	} //fopen
//
//	//ulozeni ID 
//	//(Gaussian Mixture Model - Distance based - Var Normalization
//	if (fwrite("GMM-DB-VN",sizeof(char)*9,1,fw) != 1){
//		printf("GMModel::WriteBIN : Error writing ID.\n");
//		return(1);
//	}
//	// ulozeni dimenze -- V PRIPADE PLNOKOVARIACNEHO MODELU JE DIMENZIA ZAPORNA, 
//	// INAK KLADNA!! -> koli spetnej kompatibilite \\ LMa&KW 29.8.2012
//	__int16 foo = GetFullCovStatus() ? -Dim : Dim;
//	if (fwrite(&foo,sizeof(__int16),1,fw) != 1){
//		printf("GMModel::WriteBIN : Error writing dimension.\n");
//		return(1);
//	}
//	//ulozeni normalizacniho vektoru
//	if (fwrite(NormCoef,sizeof(float) * Dim, 1, fw) != 1){
//		printf("GMModel::WriteBIN : Error writing norm. coefficients\n");
//		return(1);
//	}
//
//	//ulozeni poctu slozek
//	if (fwrite(&NumMixes,sizeof(__int16),1,fw) != 1){
//		printf("GMModel::WriteBIN : Error writing number of mixes.\n");
//		return(1);
//	}
//	//ulozeni vah jednotlivych smesi \\LMa 7.5.2009
//	if (fwrite(Weights, sizeof(float) * NumMixes, 1, fw) != 1){
//		printf("GMModel::WriteBIN : Error writing weights of mixes\n");
//		return(1);
//	}
//
//	//ulozeni jednotlivych smesi
//	for(j=0;j<NumMixes;j++) {
//		//ulozeni stredni hodnoty
//		if (fwrite(Mixes[j]->Mean, sizeof(float) * Dim, 1, fw) != 1){
//			printf("GMModel::WriteBIN : Error writing mean of %d. mixture.\n", j);
//			return(1);
//		}
//		if (fwrite(Mixes[j]->Var, sizeof(float) * DimVar, 1, fw) != 1){
//			printf("GMModel::WriteBIN : Error writing var of %d. mixture.\n", j);
//			return(1);
//		}
//	} //for j
//
//	//zavreni souboru
//	if (fclose(fw) == EOF){
//		printf("Cannot close file '%s'.\n", FileName);
//	} //close
//	return(0);
//} //WriteBIN

int GMModel::WriteTXT(const char *FileName){
	FILE *fw;
	int i;
	int j;

	//otevreni souboru
	if ((fw = fopen(FileName, "w")) == NULL){
		printf("Cannot open output file '%s'.\n", FileName);
		return(1);
	} //fopen

	//ulozeni dimenze
	fprintf(fw, "%d\n", this->Dim);
	//ulozeni normalizacniho vektoru
	for(i=0;i<Dim;i++){
		#if (DBGMM_WRITE_TXT_EXP == 1)
			fprintf(fw, "%15e\n", this->NormCoef[i]);
		#else
			fprintf(fw, "%15.10f\n", this->NormCoef[i]);
		#endif
	} //for i

	unsigned int M = 0;
	for (unsigned int m = 0; m < this->NumMixes; m++)
		M += Weights[m] > 0;

	//ulozeni poctu slozek
	fprintf(fw, "%d\n", M);
	
	//ulozeni jednotlivych smesi
	for(j=0;j<this->NumMixes;j++) {
		if (Weights[j] == 0)
			continue;

		//weight
		fprintf(fw, "%15e", this->Weights[j]);
		if(GetFullCovStatus())
			fprintf(fw, " F");
		if(this->GetMixtureOccup(j) != -1) 
			fprintf(fw, " <MIX_OCCUP> %15e", this->Mixes[j]->mixOcc);
		fprintf(fw, "\n");

		fprintf(fw, "\n");
		//mean
		for(i=0;i<Dim;i++){
			fprintf(fw, "%15e\n", this->Mixes[j]->Mean[i]);
		}
		fprintf(fw, "\n");
		
		//var
		if(GetFullCovStatus()) {
			for(i=0;i<Dim;i++){
				fprintf(fw, "%15e", this->Mixes[j]->Var[i*Dim + i]);
				for(unsigned int ii=0;ii<Dim;ii++){
					fprintf(fw, " %15e", this->Mixes[j]->Var[i*Dim + ii]);
				}	
				fprintf(fw, "\n");
			}
		}
		else {
			for(i=0;i<Dim;i++){
				fprintf(fw, "%15e\n", this->Mixes[j]->Var[i]);
			}
		}
		fprintf(fw, "\n");
	} //for j

	if (fclose(fw) == EOF){
		printf("Cannot close file '%s'.\n", FileName);
	} //close

	return(0);
} //WriteTXT



int GMModel::WriteBIN(const char *FileName){
	FILE *fw;
	int j;

	//otevreni souboru
	if ((fw = fopen(FileName, "wb")) == NULL){
		printf("Cannot open output file '%s'.\n", FileName);
		return(1);
	} //fopen

	//ulozeni ID 
	//(Gaussian Mixture Model - Distance based - Var Normalization
	if (fwrite("GMM-DB-VN",sizeof(char)*9,1,fw) != 1){
		printf("GMModel::WriteBIN : Error writing ID.\n");
		return(1);
	}
	// ulozeni dimenze -- V PRIPADE PLNOKOVARIACNEHO MODELU JE DIMENZIA ZAPORNA, 
	// INAK KLADNA!! -> koli spetnej kompatibilite \\ LMa&KW 29.8.2012
	__int16 foo = GetFullCovStatus() ? -Dim : Dim;
	if (fwrite(&foo,sizeof(__int16),1,fw) != 1){
		printf("GMModel::WriteBIN : Error writing dimension.\n");
		return(1);
	}
	//ulozeni normalizacniho vektoru
	if (fwrite(NormCoef,sizeof(float) * Dim, 1, fw) != 1){
		printf("GMModel::WriteBIN : Error writing norm. coefficients\n");
		return(1);
	}

	__int16 M = 0;
	float *ws = new float[NumMixes];
	for (unsigned int m = 0; m < NumMixes; m++) {
		if (Weights[m] > 0) {			
			ws[M] = Weights[m];
			M++;
		}
	}
	
	//ulozeni poctu slozek
	if (fwrite(&M,sizeof(__int16),1,fw) != 1){
		printf("GMModel::WriteBIN : Error writing number of mixes.\n");
		return(1);
	}

	//ulozeni vah jednotlivych smesi
	if (fwrite(ws, sizeof(float) * M, 1, fw) != 1){
		printf("GMModel::WriteBIN : Error writing weights of mixes\n");
		return(1);
	}
	
	delete [] ws;

	//ulozeni jednotlivych smesi
	for(j=0;j<NumMixes;j++) {
		if (Weights[j] == 0)
			continue;

		//ulozeni stredni hodnoty
		if (fwrite(Mixes[j]->Mean, sizeof(float) * Dim, 1, fw) != 1){
			printf("GMModel::WriteBIN : Error writing mean of %d. mixture.\n", j);
			return(1);
		}
		if (fwrite(Mixes[j]->Var, sizeof(float) * DimVar, 1, fw) != 1){
			printf("GMModel::WriteBIN : Error writing var of %d. mixture.\n", j);
			return(1);
		}
	} //for j

	//zavreni souboru
	if (fclose(fw) == EOF){
		printf("Cannot close file '%s'.\n", FileName);
	} //close
	return(0);
} //WriteBIN


int	GMModel::ReadTXT(const char *FileName){
	FILE *fr;
	int i;
	int j;
	char buffer[GMM_FILEBUFFERSIZE+1],occupInfo[GMM_FILEBUFFERSIZE+1];

	//uvolneni pameti - pripadny pozustatek pri 
	//nacitani dat do jiz existujiciho modelu
	if (Weights != NULL) free((void*) Weights);
	if (NormCoef != NULL) free((void*) NormCoef);
	if (Mixes != NULL){
		int i;
		for(i=0;i<NumMixes;i++){
			if (Mixes[i] != NULL) delete Mixes[i];
		} //for i
		free((void*) Mixes);
	}
	Weights = NULL;
	NormCoef = NULL;
	Mixes = NULL;

	//otevreni souboru
	if ((fr = fopen(FileName, "r")) == NULL){
		printf("Cannot open file '%s'.\n", FileName);
		return(1);
	} //fopen

	//nacteni dimenze
	if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
	sscanf(buffer, "%d", &Dim);

	//nacteni normalizacniho vektoru
	//alokace pameti pro pole normalizacnich koeficientu
	if ((this->NormCoef = (float*) malloc(sizeof(float)*Dim)) == NULL){
		printf("Cannot allocate memory");
		return(1);
	}
	for(i=0;i<Dim;i++){
		if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
		sscanf(buffer, "%f", &(NormCoef[i]));
	} //for i

	//nacteni poctu slozek
	if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
	sscanf(buffer, "%d", &(NumMixes));
	
	//nacteni jednotlivych smesi
	//alokace pameti pro pole vah smesi
	if ((this->Weights = (float*) malloc(sizeof(float)*NumMixes)) == NULL){
		printf("Cannot allocate memory");
		return(1);
	}
	//alokace pameti pro pole smesi
	if ((this->Mixes = (GMMMixture**) malloc(sizeof(GMMMixture*)*NumMixes)) == NULL){
		printf("Cannot allocate memory");
		return(1);
	}
	//for(i=0;i<NumMixes;i++){
	//	if ((Mixes[i] = new GMMMixture(Dim)) == NULL){
	//		printf("Cannot allocate memory.\n");
	//		return(1);
	//	}
	//} //for i

	char cF = '0';	
	for(j=0;j<this->NumMixes;j++){
		//weight & mixOccup
		float weight;
		if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
		sscanf(buffer, "%f %c", &(weight), &cF);		
		
		Mixes[j] = new GMMMixture(Dim, cF == 'F');
		if (Mixes[j] == NULL){
			printf("Cannot allocate memory.\n");
			return(1);
		}		
		Weights[j] = weight;
			
		if (cF != 'F') {
			DimVar = Dim;
			sscanf(buffer, "%f %s %f", &(weight), occupInfo, &(Mixes[j]->mixOcc));
		}
		else {
			DimVar = Dim * Dim;
			sscanf(buffer, "%f %c %s %f", &(weight), &cF, occupInfo, &(Mixes[j]->mixOcc));
		}
		
		if(strcmp(occupInfo,"<MIX_OCCUP>") != 0) {
			Mixes[j]->mixOcc = -1.0;			
		}

		//prazdna radka
		if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
		//mean
		for(i=0;i<Dim;i++){
			if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
			sscanf(buffer, "%f", &(Mixes[j]->Mean[i]));
		}
		//prazdna radka
		if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
		//var
		if(GetFullCovStatus()) {
			float foo;
			for(i = 0; i < Dim; i++) {
				if (fscanf(fr, "%f", &foo) == NULL) {return(1);}
				for(unsigned int ii = 0; ii < Dim; ii++) {
					if (fscanf(fr, "%f", &(Mixes[j]->Var[i * Dim + ii])) == NULL) {return(1);}					
				}
			}
			if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
		}
		else {
			for(i=0;i<DimVar;i++){
				if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
				sscanf(buffer, "%f", &(Mixes[j]->Var[i]));
			}
		}
		//prazdna radka
		if (fgets(buffer, GMM_FILEBUFFERSIZE, fr) == NULL) {return(1);}
	} //for j

	if (fclose(fr) == EOF){
		printf("Cannot close file '%s'.\n", FileName);
	} //close

	return(0);
} //ReadTXT



int	GMModel::ReadBIN(const char *FileName){
	FILE *fr;
	int i;
	int j;
	char ID[10];

	//uvolneni pameti - pripadny pozustatek pri 
	//nacitani dat do jiz existujiciho modelu
	if (Weights != NULL) 
		free((void*) Weights);
	if (NormCoef != NULL) 
		free((void*) NormCoef);
	if (Mixes != NULL){		
		for(int i=0;i<NumMixes;i++){
			if (Mixes[i] != NULL) 
				delete Mixes[i];
		} //for i
		free((void*) Mixes);
		Mixes = NULL;
	}

	//otevreni souboru
	if ((fr = fopen(FileName, "rb")) == NULL){
		printf("Cannot open file '%s'.\n", FileName);
		return(1);
	} //fopen

	//nacteni ID 
	//(Gaussian Mixture Model - Distance based - Var Normalization
	if (fread(ID, sizeof(char)*9, 1, fr) != 1) {return(1);}
	if (strncmp(ID, "GMM-DB-VN",9) != 0){
		printf("Incorrect file format (not GMM-DB-VN).\n");
		return(1);
	}

	//nacteni dimenze
	if (fread(&Dim, sizeof(Dim), 1, fr) != 1) {return(1);}
	if (Dim < 0) {
		Dim = -Dim;
		DimVar = Dim * Dim;
	}
	else DimVar = Dim;

	//nacteni normalizacniho vektoru
	//alokace pameti pro pole normalizacnich koeficientu
	if ((this->NormCoef = (float*) malloc(sizeof(float)*Dim)) == NULL){
		printf("Cannot allocate memory");
		return(1);
	}
	
	if (fread(NormCoef, sizeof(float)*Dim, 1, fr) != 1) 
		return(1);

	//nacteni poctu slozek
	if (fread(&NumMixes, sizeof(NumMixes), 1, fr) != 1) {return(1);}

	//nacteni vah jednotlivych smesi
	//alokace pameti pro pole vah smesi
	if ((this->Weights = (float*) malloc(sizeof(float)*NumMixes)) == NULL){
		printf("Cannot allocate memory");
		return(1);
	}
	//LMa 7.5.2009
	if (fread(Weights, sizeof(float)*NumMixes, 1, fr) != 1) {
		return(1);
	}

	//nacteni jednotlivych smesi
	//alokace pameti pro pole smesi
	if ((this->Mixes = (GMMMixture**) malloc(sizeof(GMMMixture*)*NumMixes)) == NULL){
		printf("Cannot allocate memory");
		return(1);
	}
	for(i=0;i<NumMixes;i++){
		if ((Mixes[i] = new GMMMixture(Dim, DimVar != Dim)) == NULL){
			printf("Cannot allocate memory.\n");
			return(1);
		}
	} //for i

	for(j=0;j<NumMixes;j++){
		//nacteni stredni hodnoty
		if (fread(Mixes[j]->Mean, sizeof(float)*Dim, 1, fr) != 1) {
			return(1);
		}
		//nacteni var
		if (fread(Mixes[j]->Var, sizeof(float)*DimVar, 1, fr) != 1) {
			return(1);
		}
	} //for j

	//zavreni souboru
	if (fclose(fr) == EOF){
		printf("Cannot close file '%s'.\n", FileName);
	} //close
	return(0);
} //ReadBin



unsigned int GMModel::RearrangeMixtures() {
	
	GMMMixture **nM = new GMMMixture* [NumMixes];
	float *nW = new float[NumMixes];
	
	unsigned int mnonz = 0, mz = 1;
	for (unsigned int m = 0; m < NumMixes; m++) {
		if (Weights[m] > 0) {
			nM[mnonz] = Mixes[m];
			nW[mnonz] = Weights[m];
			++mnonz;
		}
		else {
			nM[NumMixes - mz] = Mixes[m];
			nW[NumMixes - mz] = Weights[m];
			++mz;
		}
	}

	delete [] Weights;
	delete [] Mixes;

	Weights = nW;
	Mixes = nM;

	NumMixesNZW = mnonz;

	return mz - 1;
}



void GMModel::AddMixture(float *Mean, float *Var, float Weight){
	int i;
	//vektor pointru na jednotlive slozky
	GMMMixture **pom_mixes;
	if ((pom_mixes = (GMMMixture**) malloc(sizeof(GMMMixture*)*(NumMixes+1))) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}
	for(i=0;i<NumMixes;i++){
		pom_mixes[i] = Mixes[i];
	} //for i
	free((void*) Mixes);
	Mixes = pom_mixes;

	Mixes[NumMixes] = new GMMMixture(this->Dim, GetFullCovStatus());
	Mixes[NumMixes]->SetMean(Mean);
	Mixes[NumMixes]->SetVar(Var);

	//vaha
	float *pom_weights;
	if ((pom_weights = (float*) malloc(sizeof(float)*(NumMixes+1))) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}
	for(i=0;i<NumMixes;i++){
		pom_weights[i] = Weights[i];
	} //for i
	pom_weights[NumMixes] = Weight;
	free((void*) Weights);
	Weights = pom_weights;

	//zvyseni celkoveho poctu
	NumMixes++;

} //AddMixture;



unsigned int GMModel::GetBlockSize() {
	return 2*sizeof(int) + (Dim + NumMixes + NumMixes*Dim + NumMixes*DimVar)*sizeof(float);
} //GetBlockSize



unsigned int GMModel::WriteBlock(void *StartAdr) {
	long i;
	__int8 *p;

	p = (__int8*) StartAdr;

	//ulozeni dimenze
	__int16 foo = Dim;
	if (GetFullCovStatus())
		foo = -Dim;
		
	memcpy((void*) p, (void*)&foo, sizeof(__int16));
	p += sizeof(__int16);

	//ulozeni poctu slozek
	memcpy((void*) p, (void*)&NumMixes, sizeof(__int16));
	p += sizeof(__int16);

	//ulozeni vektoru vah
	memcpy((void*) p, (void*)Weights, sizeof(float)*NumMixes);
	p += sizeof(float)*NumMixes;
	
	//ulozeni normalizacnich koeficientu
	memcpy((void*) p, (void*)NormCoef, sizeof(float)*Dim);
	p += sizeof(float)*Dim;

	//ulozeni jednotlivych slozek
	for(i=0;i<NumMixes;i++){
		memcpy((void*) p, (void*)(Mixes[i]->Mean), sizeof(float)*Dim);
		p += sizeof(float)*Dim;
		memcpy((void*) p, (void*)(Mixes[i]->Var), sizeof(float)*DimVar);
		p += sizeof(float)*DimVar;
		//poznamka: Dim jako atribut slozky neni ukladano znovu
	}

	//vrati pocet zapsanych bytu
	return(p - (__int8*) StartAdr);

} //WriteBlock



unsigned int GMModel::ReadBlock(void *StartAdr) {
	long i;
	__int8 *p;
	
	//uvolneni pripadne stavajici pameti
	CleanMemory();
	
	//nacteni jednotlivych polozek
	p = (__int8*) StartAdr;

	//nacteni dimenze
	memcpy((void*)&Dim, (void*) p, sizeof(__int16));
	p += sizeof(__int16);
	
	if (Dim < 0) {
		Dim = -Dim;
		DimVar = Dim * Dim;
	}
	else DimVar = Dim;

	//nacteni poctu slozek
	memcpy((void*)&NumMixes, (void*) p, sizeof(__int16));
	p += sizeof(__int16);

	//alokace pameti zavisle na dimenzi a poctu slozek
	  //alokace pameti pro pole vah smesi
	if ((this->Weights = (float*) malloc(sizeof(float)*NumMixes)) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}
	  //alokace pameti pro pole normalizacnich koeficientu
	if ((this->NormCoef = (float*) malloc(sizeof(float)*Dim)) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}
	  //alokace pameti pro pole smesi
	if ((this->Mixes = (GMMMixture**) malloc(sizeof(GMMMixture*)*NumMixes)) == NULL){
			printf("Cannot allocate memory");
			exit(1);
	}
	  //vytvoreni jednotlivych smesi (bez inicializace vektoru)
	for(i=0;i<NumMixes;i++){
		if ((Mixes[i] = new GMMMixture(Dim, Dim != DimVar)) == NULL){
			printf("Cannot allocate memory.\n");
			exit(1);
		}
		Weights[i] = (float) ((double) 1.0/ (double) NumMixes);
	} //for i

	//nacteni vektoru vah
	memcpy((void*)Weights, (void*) p, sizeof(float)*NumMixes);
	p += sizeof(float)*NumMixes;
	
	//nacteni normalizacnich koeficientu
	memcpy((void*)NormCoef, (void*) p, sizeof(float)*Dim);
	p += sizeof(float)*Dim;

	//nacteni jednotlivych slozek
	for(i=0;i<NumMixes;i++){
		memcpy((void*)(Mixes[i]->Mean), (void*) p, sizeof(float)*Dim);
		p += sizeof(float)*Dim;
		memcpy((void*)(Mixes[i]->Var), (void*) p, sizeof(float)*DimVar);
		p += sizeof(float)*DimVar;
		////poznamka: Dim jako atribut slozky nebylo ukladano
		//Mixes[i]->Dim = this->Dim;
	}

	//vrati pocet prectenych bytu
	return(p - (__int8*) StartAdr);

} //ReadBlock


int GMModel::Load(const char *FileName, bool loadTxt)
{	
	if (loadTxt)
		return ReadTXT(FileName);
	else
		return ReadBIN(FileName);

	return(0);
} //Load



int GMModel::Save(const char *FileName, bool saveTxt)
{
	if (saveTxt)
		return WriteTXT(FileName);
	else
		return WriteBIN(FileName);

	return(0);
} //Save

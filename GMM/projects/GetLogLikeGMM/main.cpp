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

#include "tools/usefull_funcs.h"
#include "matrices/IOMats.h"
#include "trainer/OL_GMMStatsEstimator.h"
#include "trainer/OL_GMMStatsEstimator_SSE.h"
#ifdef _CUDA
#	include "trainer/OL_GMMStatsEstimator_CUDA.h"
#endif
#include "tools/OL_FileList.h"
#include "model/OL_GMModel.h"

#include <string>
#include <time.h>
#include <stdexcept>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>


using namespace std;
using namespace utilities;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

#define realT float

struct OPTIONS {
	string config_file;
	string GMM_file;

	vector<string> inputFile;
	vector<string> inputList;

	string outputDir;
	//string outputFile;
	string inExt;
	string outExt;

	int load_type;

	bool forEachVec;

	bool SSEacc;
	bool CUDAacc;
	int cudaGPUid;
	unsigned int numThreads;
	unsigned int dwnsmp;

	bool load_txt;
	bool save_txt;	
	int verbosity;
};

string error_report;


void printHead();
void printNotes();
bool LoadConfiguration (int argc, char *argv[], OPTIONS &opt);
bool checkOptions(po::variables_map &vm, OPTIONS &opt);
void loadFileLists(CFileList& files, OPTIONS& opt);

int main(int argc, char* argv[])
{
	OPTIONS opt;
	if(!LoadConfiguration(argc, argv, opt))
		return 1;

	clock_t time_b, time_e;
	time_b = clock();

	try {
		// load input lists
		CFileList files("Files");
		CFileList **fileLists;

		loadFileLists(files, opt);

		// load GMM
		GMModel GMM;
		if (GMM.Load(opt.GMM_file.c_str(), opt.load_txt) != 0) {
			error_report = string("GMM model [") + opt.GMM_file + "] not found";
			throw runtime_error(error_report.c_str());
		}
		unsigned int M = GMM.GetNumberOfMixtures();
		unsigned int D = GMM.GetDimension();

		// prepare estimator
		GMMStatsEstimator<realT> *estimator;
		if(opt.SSEacc) {
			estimator = new GMMStatsEstimator_SSE <realT>;
		}
#ifdef _CUDA
		else if(opt.CUDAacc) {
			estimator = new GMMStatsEstimator_CUDA <realT> (opt.cudaGPUid);
		}
#endif
		else {
			estimator = new GMMStatsEstimator <realT>;
		}

		if(opt.verbosity > 1)
			cout << " [" << estimator->getEstimationType().c_str() << " Estimation ON]" << endl;

		estimator->insertModel(GMM);
		estimator->_numThreads = opt.numThreads;
		estimator->_verbosity = opt.verbosity;

		char *filename;
		string outFileName;

		if(opt.verbosity > 1)
			cout << "Log-likes are going to be computed [#files = " << files.ListLength() << "]" << endl;

		float *outLLs = NULL;
		unsigned int outLLs_size = 0;

		Param param;
		files.Rewind();
		while(files.GetItemName(&filename)) {		
			
			if(param.Load(filename, opt.load_type, opt.dwnsmp) != 0) {
				cerr << "WARNING: Param file [" << filename << "] not found -> skipped!" << endl;
				continue;
			}

			unsigned int NSamples = (unsigned int) param.GetNumberOfVectors();
			unsigned int prmdim = (unsigned int) param.GetVectorDim();

			if(NSamples < 1 && opt.verbosity > 1) {
				cerr << "WARNING: Param file [" << filename << "] empty" << endl;
				continue;
			}

			if (NSamples > outLLs_size) {
				delete[] outLLs;
				outLLs = new float [NSamples];
				outLLs_size = NSamples;
			}
			estimator->compVecsLogLikeMT(*param.GetVectors(), NSamples, prmdim, outLLs);

			if(estimator->_fNumStabilityTroubles) {
				cerr << "WARNING: numerical stability troubles detected [file> " << filename << "]" << endl;
				estimator->_fNumStabilityTroubles = false;
			}

			if (opt.forEachVec) {
				string foo(filename);
				GetNewPath(outFileName, foo, opt.outputDir, opt.outExt);
				saveMat<float> (outFileName.c_str(), opt.save_txt, outLLs, 1, NSamples);
			}
			else {
				float meanLL = 0.0f;
				for (unsigned int n = 0; n < NSamples; n++)
					meanLL += outLLs[n];
				meanLL /= NSamples;
				cout << "\r" << filename << "\t" << meanLL << "\n";
			}
		}

		delete estimator;
		delete[] outLLs;
	}
	catch (bad_alloc& ba) {
		cerr << "Out of memory!" << endl;
		cerr << ba.what() << endl;
	}
	catch (exception& e) {
		cerr << e.what() << endl;		
	}
	catch (...) {
		cerr << "Unknown exception caught!" << endl << "Please, contact the developer." << endl;
	}

	time_e = clock();
	if(opt.verbosity > 1)
		cout << endl << "[Processing time = " << (time_e - time_b)/(double) CLOCKS_PER_SEC << " seconds]" << endl;

	return 0;
}



void printHead() {
	cout << "Modul name: GMM-LOG-LIKE" << endl;
	cout << "=-=-=-=-=-=-=-=-=-=-=-=-=" << endl;
	cout << "Modul version: 1.1 (May 5th, 2014)" << endl << endl;
}



void printNotes(){
	cout << endl;
	cout << " Notes: . Log-likes are computed according to the given GMM and specified files with feature vectors." << endl;
	cout << "        . If only one (mean) log-like for each file is requested, the log-likes are written" << endl;
	cout << "        on the standard output; to get rid of (possible) warnings, redirect the standard error stream" << endl;
	cout << "        to a different file and set verobsity to 0." << endl;
	cout << endl;
}



void loadFileLists(CFileList& files, OPTIONS& opt) 
{
	// check type of input -> file | direcotry | list
	if(opt.inputFile.size() > 0) {
		vector<string>::iterator it;
		for(it = opt.inputFile.begin(); it != opt.inputFile.end(); it++) {
			if(fs::is_regular_file(fs::path(*it)))
				files.AddItem((*it).c_str());
			else
				cout << "WARNING: input file " << *it << " not found" << endl;
		}
	}

	if(!opt.inputList.empty()) {
		vector<string>::iterator it;
		for(it = opt.inputList.begin(); it != opt.inputList.end(); it++) {
			if(!(*it).empty())
				ReadTestList((*it).c_str(), &files, opt.inExt.c_str());
		}
	}

	if(files.ListLength() < 1) {
		error_report = string("None suitable input files found");
		throw runtime_error(error_report.c_str());
	}
}



bool LoadConfiguration(int argc, char *argv[], OPTIONS &opt) {
	po::variables_map vm;
	po::options_description *ad_desc = NULL;
	try {		
		// parse command line
		po::options_description desc("Options-MAIN");	
		desc.add_options()
			("help,h", "display this help")

			("config,c", po::value<string>(&opt.config_file), 
			"configuration file")

			("in-file,i", po::value< vector<string> >(&opt.inputFile),
			"file(s) with feature vectors\noption may be specified several times")

			("in-list,I", po::value< vector<string> >(&opt.inputList),
			"directory with files containing feature vectors\noption may be specified several times")

			("in-GMM,g", po::value<string>(&opt.GMM_file), 
			"path to GMM")

			("for-each,f", po::value<bool>(&opt.forEachVec)->implicit_value(1)->default_value(0),
			"give log-like for each feature vector in specified files (will be stored in a directory specified by '--outDir DIR' with filename equal to the input file), otherwise the mean log-like of feature vectors from one file will be computed, and all log-likes will be stored in one file specified by '--outFile filename'")

			("outDir,o", po::value<string>(&opt.outputDir),
			"where to store files containing log-likes (needed to be specified only if --for-each option ON)")

			//("outFile,o", po::value<string>(&opt.outputFile),
			//"output file containing mean log-likes (needed to be specified only if --for-each option OFF)")

			("inExt,e", po::value<string>(&opt.inExt)->default_value(".*"), 
			"extension of input files; also a wild card '.*' may be used")

			("outExt,x", po::value<string>(&opt.outExt)->default_value(".logL"), 
			"extension of output files (if --for-each option ON)")

			("load-txt,L", po::value<bool>(&opt.load_txt)->implicit_value(1)->default_value(0),
			"load GMM from a text file")

			("save-txt,T", po::value<bool>(&opt.save_txt)->implicit_value(1)->default_value(0),
			"save log-likes into a text file (if --for-each option ON);\nBIN file: [size_x(__int32), size_y(__int32), size_x*size_y*sizeof(float)]")

			("data-T", po::value<int>(&opt.load_type)->default_value(0),
			"load input files in sves(0)/htk(1)/raw(2) format (default: 0 = SVES format)")

			("SSE", po::value<bool>(&opt.SSEacc)->implicit_value(true)->default_value(false),
			"use SSE instructions (speed boost)")
#ifdef _CUDA
			("CUDA", po::value<bool>(&opt.CUDAacc)->implicit_value(true)->default_value(false),
			"use GPU CUDA (speed boost - NVIDIA GPU needed)")

			("GPU-id,G", po::value<int>(&opt.cudaGPUid)->default_value(-1),
			"if more GPUs are available, which one should be used (0 - (#available_devices-1)); for --GPU-id=-1 the computation will be distributed among all available GPUs")
#endif
			("numThrd", po::value<unsigned int>(&opt.numThreads)->default_value(1),
			"number of CPUs used")

			("dwnsmp", po::value<unsigned int>(&opt.dwnsmp)->default_value(1),
			"down-sample factor of input feature vectors")

			("verbosity,v", po::value<int>(&opt.verbosity)->implicit_value(1)->default_value(0),
			"set verbose mode")
			;

		if(argc < 2) {	
			printHead();
			printNotes();
			cout << desc;	
			return false;
		}		
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);		

		// print help if desired
		if(vm.count("help")) {
			printHead();
			printNotes();
			cout << desc;
			return false;
		}
		
		// parse config.ini
		if(vm.count("config")) {
			ifstream ifs(opt.config_file.c_str());		
			if(ifs.fail()) {
				cout << " ERROR: LoadConfiguration(): Configuration file " << opt.config_file << " not found!" << endl;
				return false;
			}
			po::store(po::parse_config_file(ifs, desc, true), vm);
			po::notify(vm);	
			ifs.close();
		}
	}
	catch(exception &e)
	{
		cout << e.what() << "\n";
		return false;
	}		
	return checkOptions(vm, opt);
}

bool checkOptions(po::variables_map &vm, OPTIONS &opt) {
	if(vm.count("in-file") == 0 && vm.count("in-list") == 0) {
		cout <<  "LoadConfiguration(): input not specified (options -i | -I)!" << endl;
		cout <<  "                     See help (option -h) for more options" << endl;
		return false;
	}

	if(vm.count("in-GMM") == 0) {
		cout <<  "LoadConfiguration(): \"in-GMM\" not specified!" << endl;
		cout <<  "       See help (option -h) for more options" << endl;
		return false;
	}

	if(opt.forEachVec && vm.count("outDir") == 0) {
		cout <<  "LoadConfiguration(): \"outDir\" has to be specified!" << endl;
		cout <<  "         See help (option -h) for more options" << endl;
		return false;
	}

	//if(!opt.forEachVec && vm.count("outFile") == 0) {
	//	cout <<  "LoadConfiguration(): \"outFile\" has to be specified!" << endl;
	//	cout <<  "         See help (option -h) for more options" << endl;
	//	return false;
	//}

	return true;
}


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
#   include "general/my_inttypes.h"
#endif

#include "general/Exception.h"
#include "tools/OL_FileList.h"
#include "tools/OL_MLFParser.h"
#include "tools/usefull_funcs.h"
#include "param/OL_Param.h"
#include "trainer/OL_ModelAdapt.h"
#include "model/OL_GMModel.h"

#include <new>
#include <time.h>
#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <list>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>


using namespace bmssr;
using namespace stdext;
using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;


struct OPTIONS {
	string config_file;
	string UBM_file;
	string UBM_dir;
	string UBM_ext;
	vector<string> inputFile;
	vector<string> inputFileClasses;
	vector<string> inputList;
	string outputFile;
	string inExt;
	string outExt;
	string outDir;
	bool trainFromScratch;	
	bool fullCovGMM;
	bool mlf_input;
	bool class_input;
	bool eachFile1Model;
	bool continuos_mode;
	int load_type;
	bool load_txt;
	bool save_txt;
	bool save_in_stat_format;
	bool noInputModel;
	unsigned int nummix;
	unsigned int prmSamplePeriod;
	unsigned int dwnsmp;
	int verbosity;
};
void printHead();
bool checkOptions (po::variables_map &vm, OPTIONS &opt);
bool loadConfiguration (int argc, char *argv[], OPTIONS &opt, CModelAdapt::OPTIONS &ad_options);
void loadFileLists (CFileList& files, OPTIONS& opt);
void loadFileLists (hash_map <string, hash_map < string, list < vector <unsigned int> > > >& class2frames, 
					list <string>& lmodels, OPTIONS& opt);
void loadFileLists (list <string>& lmodels, 
					hash_map < string, CFileList* >& classList, OPTIONS& opt);
bool fillAdaptationWithData (CModelAdapt& modelAdapt, CFileList& files, OPTIONS& opt);
bool fillAdaptationWithData (CModelAdapt& modelAdapt, hash_map < string, list < vector <unsigned int> > >& frameslist, OPTIONS& opt);
GMModel* loadUBM (string& modelname, OPTIONS& opt);


int main(int argc, char* argv[])
{			
	try {
		// get configuration
		OPTIONS opt;
		CModelAdapt::OPTIONS ad_opt;
		if(!loadConfiguration(argc, argv, opt, ad_opt))
			return RETURN_FAIL;		

		if(opt.verbosity)
			printHead();

		clock_t time_b, time_e;
		if(opt.verbosity)  {
			time_b = clock();
			cout << " Reading train data ... ";
		}

		// read training list content
		list <string> lmodels;
		hash_map <string, CFileList*> fileLists;
		hash_map <string, hash_map < string, list < vector <unsigned int> > > > class2frames;

		if (opt.mlf_input) {
			loadFileLists(class2frames, lmodels, opt);
			if(class2frames.empty()) {
				printf("Non input files found!\n");
				return RETURN_FAIL;
			}
		}
		else if (opt.class_input) {
			loadFileLists(lmodels, fileLists, opt);
			if(lmodels.empty()) {
				printf("Non input files found!\n");
				return RETURN_FAIL;
			}
		}
		else {
			CFileList* files = new CFileList("Files");			
			loadFileLists(*files, opt);
			if(files->ListLength() < 1) {
				printf("Non input files found!\n");
				return RETURN_FAIL;
			}
			if(opt.eachFile1Model) {
				char *filename;
				files->Rewind();
				while(files->GetItemName(&filename)) {
					CFileList* file = new CFileList("file");
					file->AddItem(filename);

					lmodels.push_back(fs::basename(fs::path(filename)));
					fileLists[lmodels.back()] = file;
				}
				delete files;
				//opt.UBM_dir.clear();
			}
			else {
				lmodels.push_back(fs::basename(fs::path(opt.outputFile)));
				fileLists[lmodels.front()] = files;
			}
		}

		if(opt.verbosity) 
			cout << " done" << endl;
		
		// adaptation BEGIN -----------------------------------		
		for (list <string>::iterator it = lmodels.begin(); it != lmodels.end(); it++) 
		{			
			string modelname;
			if (opt.mlf_input || opt.class_input || opt.eachFile1Model)
				modelname = opt.outDir + "/" + *it + opt.outExt;				
			else
				modelname = opt.outputFile; // opt.outputFile != *it

			GMModel* UBM = NULL;
			if (!opt.noInputModel) {
				// load UBM
				if(opt.verbosity)
					cout << " [MODEL: " << modelname << "]\n Reading UBM model ... ";

				UBM = loadUBM(*it, opt);

				if(opt.verbosity)
					cout << " done\n Loading data ... ";
			}
			else if(opt.verbosity)
					cout << "\n Loading data ... ";
			
			clock_t time_bd, time_ed;
			if(opt.verbosity)
				time_bd = clock();

			// adapt the UBM
			CModelAdapt modelAdapt;
			modelAdapt.SetOptions(ad_opt);			

			bool data_inserted = true;
			if (opt.mlf_input) {
				if (opt.continuos_mode)
					modelAdapt.InsertData(&class2frames[*it]);
				else 
					data_inserted = fillAdaptationWithData(modelAdapt, class2frames[*it], opt);
			}
			else {
				if (opt.continuos_mode)
					modelAdapt.InsertData(fileLists[*it]);
				else
					data_inserted = fillAdaptationWithData(modelAdapt, *fileLists[*it], opt);
			}

			if(!data_inserted) {
				cerr << " WARNING: none data, adaptation skipped!" << endl;
				continue;
			}
					
			if(opt.verbosity) {
				time_ed = clock();
				cout << " done in " << (time_ed - time_bd)/(double) CLOCKS_PER_SEC << "s" << endl << " TRAINING -> ";
			}

			if (!opt.trainFromScratch)
				modelAdapt.AdaptModel(UBM);
			else 
				modelAdapt.TrainFromScratch(opt.nummix, opt.fullCovGMM, 
				                            UBM, modelname.c_str(), opt.save_txt);

			if(opt.verbosity)
				cout << " done" << endl << " Saving model ... ";	

			modelAdapt.Save(modelname.c_str(), opt.save_txt, opt.save_in_stat_format);

			if(opt.verbosity) 
				cout << " done" << endl << endl;
		}

		// adaptation END -------------------------------------
		for (hash_map <string, CFileList*>::iterator it = fileLists.begin(); it != fileLists.end(); it++)
			delete (*it).second;
		
		if(opt.verbosity) {
			time_e = clock();
			cout << "[Processing time = " << (time_e - time_b)/(double) CLOCKS_PER_SEC << " seconds]" << endl;
		}
	}
	catch (bad_alloc& a) {
		cout << "ERROR: not enough memory.\n" << a.what() << endl;
	}
	catch (exception& e) {
		cout << "ERROR: " << e.what() << endl;
	}
	catch (Exception* e) {
		e->Print();
		cout << endl;
	}
	catch (...) {
		cout << "ERROR: Unknown error occured" << endl;
	}
	
	return 0;
}



void printHead() {
	cout << "Modul name: GMM-ESTIMATOR" << endl;
	cout << "=-=-=-=-=-=-=-=-=-=-=-=-=" << endl;
	cout << "Modul version: 5.2 (May 5th, 2014)" << endl << endl;
}



bool loadConfiguration(int argc, char *argv[], OPTIONS &opt, CModelAdapt::OPTIONS &ad_opt) {
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
			"prm file(s) with feature vectors to be adapted, or \nMLF file -- also the -S option has to be set in order to accept MLF input,\noption may be specified several times")

			("in-list,I", po::value< vector<string> >(&opt.inputList),
			"directory/list with paths to feature vectors to be adapted,\noption may be specified several times")

			("model-file-mode", po::value<bool>(&opt.eachFile1Model)->implicit_value(1)->default_value(0),
			"a model is created for each file specified in 'in-list' and/or 'in-file'; models are stored in the directory specified by option 'outDir' and have an extension specified by option 'outExt'")

			("in-classes,S", po::value< vector<string> >(&opt.inputFileClasses),
			"file containing list of classes to be adapted, may be specified several times, two modes are available: MLF mode | CLASS mode\n. [MLF mode] -- one has to insert MLF file (option -i) with time labels, specify data directory (not mandatory, option -I, e.g. -I dir/myDir), which is added as a prefix to filenames present in the MLF, and set the extension of input files (mandatory, option -e); htk files do containalso the sample period, for other types of input file formats one has to specify the sample period manually (see option --sample-period below); i-th line syntax of the 'in-classes' input file: class_i_ID element_1 element_2 ... element_N \na model (with filename 'class_i_ID') is then created for each set of elements, e.g. if MLF contains time intervals of phonemes present in individual prm files, then one can train two models: 'silence' and 'vocals', for this purpose two lines have to be specified in the 'MLF_classes.txt':\nsilence _sil_ _lb_ _ns_\nspeech a e i o u A E I O U\n . [CLASS mode] -- input file 'in-classes' has to contain information on classes, i-th line syntax is the same as for MLF mode, however 'element_i' denotes now a prm filename (without extension, extension specified through -e), path to the prm can be given directly in the input class-file, e.g.: class_i_ID path1/path2/element_1 ... path3/path4/element_N\nor specified using option -I dir/myDir (otherwise options -I and -i have not to be given); in this mode one model (with filename 'class_i_ID') is trained from several files; Note 1: the distinction between MLF and CLASS mode is made by specifying option -i; Note 2: option -I can be  specified only once; Note 3: output models are stored in directory specified by option -D and have an extension specified by option -x")

			("sample-period", po::value<unsigned int>(&opt.prmSamplePeriod),
			"(MLF input & SVES prm - see option '--data-T') set feature sample-period for SVES files (not required for htk files) in order to derive samples from time labels occurring in the MLF")

			("outExt,x", po::value<string>(&opt.outExt)->default_value(".gmm"), 
			"extension of output GMMs if more than one output model is assumed")

			("outDir,D", po::value<string>(&opt.outDir)->default_value("./"),
			"output directory used to store output GMMs if more than one output model is assumed or model-file-mode is on")

			("in-UBM,u", po::value<string>(&opt.UBM_file), 
			"path to Universal Background Model (UBM); if not specified (along with 'in-dir-UBM') the model will be trained from scratch;\nif specified along with the option '--nummix', this model will be used for the initialization and its Gaussians will be further divided until specified number of Gaussians is reached")

			("in-dir-UBM,U", po::value<string>(&opt.UBM_dir), 
			"path to directory containing initial model(s) (UBMs) to be adapted - if 'model-file-mode' is switched on, for each specified prm file a model with the same filename will be searched in 'in-dir-UBM' and used as the UBM to be adapted (i.e. different UBM for each input file); in the CLASS or MLF mode each model is used as initial model for one class;\nbasename of model(s) should equal to 'class_i_ID' (given in the classes file (see option -S)) or to the basename of the outfile specified by option -o;\nif option not given, option -u is assumed as UBM input (one UBM for all the classes); note: only one of the options 'in-UBM' or 'in-dir-UBM' can be specified at a time")

			("full-cov", po::value<bool>(&opt.fullCovGMM)->implicit_value(1)->default_value(0),
			"if a model trained from scratch should have full covariances")

			("nummix,g", po::value<unsigned int>(&opt.nummix)->default_value(0),
			"specify number of Gaussians when model is trained from scratch;\n can be specified along with initial GMM given by the option --in-UBM or --in-dir-UBM in order to split Gaussians of an existing model")

			("inExt-gmm,E", po::value<string>(&opt.UBM_ext)->default_value(".gmm"),
			"extension of models in the directory 'in-dir-UBM'")

			("outFile,o", po::value<string>(&opt.outputFile),
			"(not for CLASS or MLF) where to store the adapted model or transformation matrix")

			("file-by-file,f", po::value<bool>(&opt.continuos_mode)->implicit_value(1)->default_value(0),
			"accumulate statistics file by file from HDD, do not store all the feature vectors in the memory")

			("inExt,e", po::value<string>(&opt.inExt), 
			"extension of input prm files if input list is a directory (default '.*', in CLASS | MLF mode the default is '.prm')")

			("data-T", po::value<int>(&opt.load_type)->default_value(0),
			"load prm files in sves(0)/htk(1)/raw(2) format (default: 0 = SVES format)")

			("load-txt,L", po::value<bool>(&opt.load_txt)->implicit_value(1)->default_value(0),
			"load UBM from a text file")

			("save-txt,T", po::value<bool>(&opt.save_txt)->implicit_value(1)->default_value(0),
			"save adapted model into a text file")

			("save-in-stat-form", po::value<bool>(&opt.save_in_stat_format)->implicit_value(1)->default_value(0),
			"save adapted model in format intended for statistics; note that also the MLLR transformation matrix can be saved in this format, rows are stored as stats.mean and if mllr-Qval is true than stats.mixProb are the values of the mllr criterion for each row of W, otherwise stats.mixProb=1")

			("dwnsmp", po::value<unsigned int>(&opt.dwnsmp)->default_value(1),
			"(not for MLF input) set feature down-sample factor")

			("verbosity,v", po::value<int>(&opt.verbosity)->implicit_value(1)->default_value(0),
			"set verbose mode")
			;
		ad_desc = CModelAdapt::GetOptions(ad_opt);
		desc.add(*ad_desc);

		if(argc < 2) {			
			cout << desc;	
			delete ad_desc;
			return false;
		}		
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);		

		// print help if desired
		if(vm.count("help")) {
			cout << desc;
			delete ad_desc;
			return false;
		}
		
		// parse config.ini
		if(vm.count("config")) {
			ifstream ifs(opt.config_file.c_str());		
			if(ifs.fail()) {
				cout << " ERROR: LoadConfiguration(): Configuration file " << opt.config_file << " not found!" << endl;
				delete ad_desc;
				return false;
			}
			po::store(po::parse_config_file(ifs, desc, true), vm);
			po::notify(vm);	
			ifs.close();
		}
		delete ad_desc;
	}
	catch(exception &e)
	{
		cout << e.what() << "\n";
		delete ad_desc;
		return false;
	}		

	opt.trainFromScratch = opt.nummix > 0;

	ad_opt._load_type = opt.load_type;
	ad_opt._verbosity = opt.verbosity;
	ad_opt._dwnsmp = opt.dwnsmp;
	ad_opt._prmSamplePeriod = opt.prmSamplePeriod;
	ad_opt._fromScratch = opt.trainFromScratch;

	return checkOptions(vm, opt);
}



bool checkOptions(po::variables_map &vm, OPTIONS &opt) {
	
	opt.mlf_input = opt.class_input = false;
	if(vm.count("in-classes")) {		
		if(vm.count("in-file"))
			opt.mlf_input = true;
		else
			opt.class_input = true;
	}

	if(!opt.mlf_input && !opt.class_input) {
		if(vm.count("in-file") == 0 && vm.count("in-list") == 0) {
			cout <<  "LoadConfiguration(): input not specified (options -i | -I)!" << endl;
			cout <<  "                     See help (option -h) for more details" << endl;
			return false;
		}

		if(!opt.eachFile1Model) {
			if(vm.count("outFile") == 0) {
				cout <<  "LoadConfiguration(): \"outFile\" not specified!" << endl;
				cout <<  "         See help (option -h) for more details" << endl;
				return false;
			}
		}
	}

	if((opt.mlf_input || opt.class_input) && opt.inputList.size() > 1) {
		cout <<  "LoadConfiguration(): \"in-list\" in MLF | CLASS mode may be specified only once!" << endl;
		cout <<  "		               See help (option -h) for more details" << endl;
		return false;
	}

	if (vm.count("in-UBM") == 0 && vm.count("in-dir-UBM") == 0)
		opt.noInputModel = true;
	else opt.noInputModel = false;

	if (vm.count("in-UBM") != 0 && vm.count("in-dir-UBM") != 0) {
		cout <<  "LoadConfiguration(): Both options \"in-UBM\" and \"in-dir-UBM\" specified!" << endl;
		cout <<  "                     Only one of these options can be specified at once!" << endl;
		cout <<  "		               See help (option -h) for more details" << endl;
		return false;
	}
		
	if(opt.trainFromScratch && opt.nummix < 1) {		
		cout <<  "LoadConfiguration(): None input model specified (\"inUBM\" or \"in-dir-UBM\") and \"nummix\" < 1!" << endl;
		cout <<  "                     In order to train a new model from scratch specify \"nummix\" (>0) or insert input GMM/UBM to be adapted!" << endl;
		cout <<  "		               See help (option -h) for more details" << endl;
		return false;
	}

	if(opt.mlf_input && (opt.load_type != HTK_PRM_IN) && vm.count("sample-period") == 0) {
		cout <<  "LoadConfiguration(): you have to specify \"sample-period\" for SVES files!" << endl;
		cout <<  "                     See help (option -h) for more details" << endl;
		return false;
	}

	if(vm.count("inExt") == 0) {
		if (opt.mlf_input) {
			cout <<  "LoadConfiguration(): you have to specify \"inExt\" for MLF mode!" << endl;
			cout <<  "                     See help (option -h) for more details" << endl;
			return false;
		}
		else if(!opt.class_input)
			opt.inExt = string(".*");
	}

	// following lines must not be uncommented!
	//if (!opt.inputList.empty() && opt.inputList.at(opt.inputList.size() - 1) != '/')
	//	opt.inputList += '/'; // WHAT IF INPUTLIST IS A FILE! ;)

	return true;
}



void loadFileLists (hash_map <string, hash_map < string, list < vector <unsigned int> > > >& class2frames, 
					list <string>& lmodels, OPTIONS& opt) 
{
	hash_map <string, list <string> > elem2class;

	for (vector<string>::iterator it = opt.inputFileClasses.begin(); it != opt.inputFileClasses.end(); it++) 
		MLF_Parser::parseClasses ((*it).c_str(), lmodels, elem2class); 

	for (vector<string>::iterator it = opt.inputFile.begin(); it != opt.inputFile.end(); it++) 
	{
		if(fs::is_regular_file(fs::path(*it)))
			MLF_Parser::parseMLF ((*it).c_str(), elem2class, class2frames);
		else
			cerr << "WARNING: input file " << *it << " not found" << endl;
	}

	// add full path to htk files
	string prefix;
	if(!opt.inputList.empty() && !opt.inputList.at(0).empty())
		prefix = opt.inputList.at(0) + '/'; 

	hash_map <string, hash_map < string, list < vector <unsigned int> > > > foo;
	for (list<string>::iterator it = lmodels.begin(); it != lmodels.end(); it++) 
	{
		if (class2frames.find(*it) == class2frames.end())
			continue;

		hash_map < string, list < vector <unsigned int> > >::iterator it1;
		for (it1 = class2frames[*it].begin(); it1 != class2frames[*it].end(); it1++) {
			string newFilename = prefix + (*it1).first + opt.inExt; 			
			foo[*it][newFilename] = class2frames[*it][(*it1).first];
		}	
	}
	class2frames.swap(foo);
}



void loadFileLists(CFileList& files, OPTIONS& opt) {

	// check type of input -> file | direcotry | list
	if(opt.inputFile.size() > 0) {
		vector<string>::iterator it;
		for(it = opt.inputFile.begin(); it != opt.inputFile.end(); it++) {
			if(fs::is_regular_file(fs::path(*it)))
				files.AddItem((*it).c_str());
			else
				cerr << "WARNING: input file " << *it << " not found" << endl;
		}
	}

	if(!opt.inputList.empty()) {
		vector<string>::iterator it;
		for(it = opt.inputList.begin(); it != opt.inputList.end(); it++) {
			if(!(*it).empty())
				ReadTestList((*it).c_str(), &files, opt.inExt.c_str());
		}
	}
}



bool fillAdaptationWithData (CModelAdapt& modelAdapt, hash_map < string, list < vector <unsigned int> > >& frameslist, OPTIONS& opt) 
{
	Param param;
	bool data_IN = false;

	if(opt.load_type != HTK_PRM_IN)
		param.SetSamplePeriod(opt.prmSamplePeriod);

	hash_map < string, list < vector <unsigned int> > >::iterator it;
	for (it = frameslist.begin(); it != frameslist.end(); it++) {

		if(param.Load ((*it).first.c_str(), frameslist[(*it).first], opt.load_type) != 0) {
			if(opt.verbosity > 1)
				cout << "File " << (*it).first << " could not be opened -> skipped" << endl;
			continue;
		}
		
		modelAdapt.InsertData(param, true);
		param.CleanMemory(true);
		data_IN = true;
	}
	//if(!data_IN)
	//	throw runtime_error("None param files found");
	return data_IN;
}



bool fillAdaptationWithData (CModelAdapt& modelAdapt, CFileList& files, OPTIONS& opt) {
	bool data_IN = false;
	char *filename;
	
	Param param;
	files.Rewind();		
	while(files.GetItemName(&filename))
	{
		//if(!fs::is_regular_file(fs::path(filename)))
		//	continue;

		if(param.Load (filename, opt.load_type, opt.dwnsmp) != 0) {
			if (opt.verbosity > 1)
				cout << "File " << filename << " could not be opened -> skipped" << endl;
			continue;
		}

		modelAdapt.InsertData(param, true);
		param.CleanMemory(true);
		data_IN = true;
	}

	//if(!data_IN)
	//	throw runtime_error("None param files found");
	return data_IN;
}



GMModel* loadUBM (string& modelname, OPTIONS& opt) {
	static bool singlUBMloaded = false;
	static GMModel singlUBM, UBM;
	
	string ubmfile;
	if (!opt.UBM_dir.empty()) 
	{	
		ubmfile = opt.UBM_dir + "/" + modelname + opt.UBM_ext;
		if (UBM.Load(ubmfile.c_str(), opt.load_txt) == 0)			
			return &UBM;
	}

	if (singlUBMloaded)
		return &singlUBM;
	
	if (!opt.UBM_file.empty() && singlUBM.Load(opt.UBM_file.c_str(), opt.load_txt) == 0) 
	{
		singlUBMloaded = true;
		return &singlUBM;
	}

	string error_report = string("UBM failed to load: [") + opt.UBM_file + "] | [" + ubmfile + "]";
	throw runtime_error(error_report.c_str());
}



void loadFileLists (list <string>& lmodels, 
					hash_map < string, CFileList* >& classList, OPTIONS& opt) 
{	

	vector<string>::iterator it;
	for (it = opt.inputFileClasses.begin(); it != opt.inputFileClasses.end(); it++)  {
		ifstream file((*it).c_str());
		if(file.fail()) {
			string error_report = string("loadFileLists(): file ") + (*it) + " could not be opened for reading";
			throw runtime_error(error_report.c_str());
		}

		boost::escaped_list_separator<char> e_l_s (string(), string(", ;:*!\n\r\t"), string("\""));

		file.exceptions(ifstream::badbit);
		try {
			string prefix;
			if(!opt.inputList.empty() && !opt.inputList.at(0).empty())
				prefix = opt.inputList.at(0) + '/'; 

			string line, class_id;
			while(getline(file, line)) {
				if (line.empty() || line.at(0) == '\r')
					continue;

				boost::tokenizer< boost::escaped_list_separator<char> > tok(line, e_l_s);
				boost::tokenizer< boost::escaped_list_separator<char> >::iterator beg = tok.begin();
				if (beg == tok.end() || (*beg).empty())
					continue;
				class_id = fs::basename(fs::path(*beg));

				CFileList* flist = new CFileList("files");
				for(beg = ++beg; beg != tok.end(); ++beg) {
					if ((*beg).empty())
						continue;
					string prmfile = prefix;
					if (!opt.inExt.empty())
						prmfile += fs::change_extension(fs::path(*beg), opt.inExt).string();
					else
						prmfile += *beg;

					flist->AddItem(prmfile.c_str());
				}
				if(flist->ListLength() > 0) {
					classList[class_id] = flist;
					lmodels.push_back(class_id);
				}
				else 
					delete flist;
			}
		}
		catch (ifstream::failure& f) {		
			file.close();

			string error_report = string("loadFileLists(): Error occured when reading file ") + (*it) + "\n\t" + f.what();
			throw runtime_error(error_report.c_str());
		}	

		file.close();
	}
}

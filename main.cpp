#include"code\PBAS-\FileHandler.h"
#include<opencv2\opencv.hpp>
#include<iostream>
#include "code\PBAS-\FrameProcessor.h"
#include"code\PBAS-\PBASFrameProcessor.h"
#include<thread>

//#define _DATASET
//#define _FOLDER
//#define _CAMERA

int main(int argc, char** argv)
{
	// PBAS Parameter Settings
	int N = 35;
	int minHits = 2;
	double alpha = 10;
	double beta = 1;

	double defaultR = 18;
	double RScale = 5;
	double RIncDec = 0.05;

	int defaultSubsampling = 16;
	double subsamplingIncRate = 1;
	double subsamplingDecRate = 0.05;
	int subsamplingLowerBound = 2;
	int subsamplingUpperBound = 200;


#ifdef _DATASET

	CV_Assert(argc == 3);
	std::string data_root = argv[1], result_root = argv[2];
	const int ROOT_DEPTH = 2;
	const int THREAD_NUM = 10;
	

	std::vector<FileHandler> f_obj_arr;
	for (int i = 0; i <THREAD_NUM; ++i)
	{
		if(i==0)
			f_obj_arr.push_back(FileHandler(data_root, result_root, ROOT_DEPTH, true));
		else
			f_obj_arr.push_back(FileHandler(data_root, result_root, ROOT_DEPTH, false));
		
		f_obj_arr[i].setDisplayFlag(true);
	}


	std::vector<PBASFrameProcessor*> frame_proc_arr(THREAD_NUM);
	for (int i = 0; i < frame_proc_arr.size(); ++i)
	{
		frame_proc_arr[i] = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
	}


	int remainder = f_obj_arr.at(0).getFolderCount() % THREAD_NUM;
	std::vector<std::string> inputFolders = f_obj_arr.at(0).getInputFolders();
	std::vector<std::string> outputFolders = f_obj_arr.at(0).getOutputFolders();
	std::thread thread_arr[THREAD_NUM];
	
	
	for (int it = 0; it < floor(f_obj_arr.at(0).getFolderCount() / THREAD_NUM); ++it)
	{

		for (int k = 0; k < THREAD_NUM; ++k)
		{
			thread_arr[k] = std::thread(static_cast<bool (FileHandler::*) (std::string, std::string, PBASFrameProcessor*)>(&FileHandler::process_folder<PBASFrameProcessor>),
										&f_obj_arr[k], inputFolders[it * THREAD_NUM + k], outputFolders[it * THREAD_NUM + k], frame_proc_arr[k]);
		}
		
		for (int k = 0; k < THREAD_NUM; ++k)
		{
				thread_arr[k].join();
		}
		
	}


	int divisor = floor(f_obj_arr.at(0).getFolderCount() / THREAD_NUM);
	for (int i = 0; i < remainder; ++i)
	{
		thread_arr[i] = std::thread(static_cast<bool (FileHandler::*) (std::string, std::string, PBASFrameProcessor*)>(&FileHandler::process_folder<PBASFrameProcessor>),
			&f_obj_arr[i], inputFolders[divisor * THREAD_NUM + i], outputFolders[divisor * THREAD_NUM + i], frame_proc_arr[i]);
		
		std::cout << divisor * THREAD_NUM + i << std::endl;
	}
	
	for (int k = 0; k < remainder; ++k)
	{
		thread_arr[k].join();
	}
	std::cout << remainder << std::endl;

#endif


#ifdef _FOLDER
	const std::string inputFolder = "";
	const std::string outputFolder = "";
	PBASFrameProcessor

#endif //  _FOLDER


#ifdef _CAMERA

#endif //  _FOLDER








	return 0;
}
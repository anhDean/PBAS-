#include"FileHandler.h"
#include<opencv2\opencv.hpp>
#include<iostream>
#include "FrameProcessor.h"
#include"PBASFrameProcessor.h"
#include<thread>
#include"LBSP.h"
#include "BackgroundFeature.h"
#include "MoGSegmenter.h"
#include "MoGFrameProcessor.h"
#include "ColourFeature.h"
#include "PBASFeature.h"

#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__<< "(" << __LINE__ << ")" << (x) << std::endl
#else
#define LOG_MESSAGE(x)
#endif

//#define _DATASET
#define _FOLDER
//#define _CAMERA
//#define _PROTOTYPING

int main(int argc, char** argv)
{
	// PBAS Parameter Settings using macro trick
#define DEFINE_FRAMEPROCESSORPARAMS(type, name, value) type name = value;
	#include "FrameProcessorParams.h"
	#undef DEFINE_FRAMEPROCESSORPARAMS

#ifdef _DATASET
	CV_Assert(argc == 3);
	std::string data_root = argv[1], result_root = argv[2];
	int x = 10;
	const int ROOT_DEPTH = 2;
	const int THREAD_NUM = 31;
	std::vector<FileHandler> f_obj_arr;
	for (int i = 0; i <THREAD_NUM; ++i)
	{
		if(i==0)
			f_obj_arr.push_back(FileHandler(data_root, result_root, ROOT_DEPTH, true)); // true: removes all folders in output root
		else
			f_obj_arr.push_back(FileHandler(data_root, result_root, ROOT_DEPTH, false));
		
		//f_obj_arr[i].setDisplayFlag(true);
	}
	std::vector<FrameProcessor*> frame_proc_arr(THREAD_NUM);
	for (int i = 0; i < frame_proc_arr.size(); ++i)
	{
		frame_proc_arr[i] = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
	}


	int remainder = f_obj_arr.at(0).getFolderCount() % THREAD_NUM;
	std::vector<std::string> inputFolders = f_obj_arr.at(0).getInputFolders();
	std::vector<std::string> outputFolders = f_obj_arr.at(0).getOutputFolders();
	std::thread thread_arr[THREAD_NUM];
	
	int t_init = cv::getTickCount();
	for (int it = 0; it < floor(f_obj_arr.at(0).getFolderCount() / THREAD_NUM); ++it)
	{
		for (int k = 0; k < THREAD_NUM; ++k)
		{
			thread_arr[k] = std::thread(static_cast<bool (FileHandler::*) (std::string, std::string, FrameProcessor*)>(&FileHandler::process_folder),
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
		thread_arr[i] = std::thread(static_cast<bool (FileHandler::*) (std::string, std::string, FrameProcessor*)>(&FileHandler::process_folder),
			&f_obj_arr[i], inputFolders[divisor * THREAD_NUM + i], outputFolders[divisor * THREAD_NUM + i], frame_proc_arr[i]);
		
		std::cout << divisor * THREAD_NUM + i << std::endl;
	}
	
	for (int k = 0; k < remainder; ++k)
	{
		thread_arr[k].join();
	}

	for (int k = 0; k < frame_proc_arr.size(); ++k)
	{
		delete frame_proc_arr[k];
	}

	std::cout << "Time needed for processing: " << ((cv::getTickCount() - t_init) / cv::getTickFrequency()) / 60 << "min" << std::endl;

#endif
#ifdef _FOLDER
	/*
	// USE THIS CODE TO GENERATE EXECUTEABLE THAT IS COMPLIANT WITH CHANGEDETECTION EVALUTATION CODE

	const std::string inputFolder = argv[1];
	const std::string outputFolder = argv[2];
	
	FrameProcessor* processor = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
	FileHandler fh;

	fh.setInputPrefix("in");
	fh.setInputSuffix(".jpg");

	fh.setOutputPrefix("bin");
	fh.setOutputSuffix(".png");

	fh.process_folder(inputFolder, outputFolder, processor);
	delete processor;
	*/

	const std::string inputFolder = "E:\\Datasets\\datasets2012\\dataset\\dynamicBackground\\fountain01\\input";
	const std::string outputFolder = "E:\\Test";
	FrameProcessor* processor = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
	//FrameProcessor* processor = new MoGFrameProcessor(4, PBASFeature::NUM_FEATURES);
	FileHandler fh;
	fh.setDisplayFlag(true);
	fh.process_folder(inputFolder, outputFolder, processor);

	delete processor;



#endif
#ifdef _CAMERA
	FrameProcessor *processor = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
	
	cv::VideoCapture cap(0);// open default camera
	if (!cap.isOpened())
		return -1;

	cv::Mat frame, out;
	while (true)
	{
		cap >> frame; // get a new frame from camera
		processor->process(frame, out);
		cv::cvtColor(frame, frame, CV_BGR2GRAY);
		out = out / 255; // get binary map {0,1}
		cv::imshow("output", frame.mul(out)); // mask input frame with segmentation map
		cv::imshow("input", frame);
		if (cv::waitKey(30) >= 0) break;
	}
#endif

#ifdef _PROTOTYPING
	// block for experimenting, design testing
	PBASFeature x(1, 2, 3, 0.8);
	PBASFeature y(x * 2);
	PBASFeature diff =   x * 2 - y;

	std::cout << x[0]<< std::endl;
	std::cout << x[1] << std::endl;
	std::cout << x[2] << std::endl;
	std::cout << x[3] << std::endl;

	std::cout << std::endl;

	std::cout << y[0] << std::endl;
	std::cout << y[1] << std::endl;
	std::cout << y[2] << std::endl;
	std::cout << y[3] << std::endl;

	std::cout << std::endl;

	std::cout << diff[0] << std::endl;
	std::cout << diff[1] << std::endl;
	std::cout << diff[2] << std::endl;
	std::cout << diff[3] << std::endl;

	std::cout << std::endl;

	std::cout<< PBASFeature::calcDistance(x, y) << std::endl;


#endif

return 0;
}
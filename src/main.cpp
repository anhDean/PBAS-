#include"FileHandler.h"
#include<opencv2\opencv.hpp>
#include<iostream>
#include "FrameProcessor.h"
#include"PBASFrameProcessor.h"
#include<thread>
#include"LBSP.h"

#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__<< "(" << __LINE__ << ")" << (x) << std::endl
#else
#define LOG_MESSAGE(x)
#endif

//#define _DATASET
//#define _FOLDER
//#define _CAMERA
#define _PROTOTYPING

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
	const int THREAD_NUM = 31;
	std::vector<FileHandler> f_obj_arr;
	for (int i = 0; i <THREAD_NUM; ++i)
	{
		if(i==0)
			f_obj_arr.push_back(FileHandler(data_root, result_root, ROOT_DEPTH, true));
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
			thread_arr[k] = std::thread(static_cast<bool (FileHandler::*) (std::string, std::string, FrameProcessor*)>(&FileHandler::process_folder<FrameProcessor>),
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
		thread_arr[i] = std::thread(static_cast<bool (FileHandler::*) (std::string, std::string, FrameProcessor*)>(&FileHandler::process_folder<FrameProcessor>),
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

	std::cout << "Time needed for processing: " << (cv::getTickCount() - t_init) / (cv::getTickFrequency() * 60) << "min" << std::endl;

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


	const std::string inputFolder = "";
	const std::string outputFolder = "";
	FrameProcessor* processor = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
	FileHandler fh;

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
	cv::Mat test_frame = cv::imread("C:\\Users\\Public\\Pictures\\Sample Pictures\\Koala.jpg");
	std::string windowName = "test image";

	LBSP l1(test_frame);
	l1.displayLBSPXY(10, 10);
	LBSP::displayPatchXY(test_frame, 10, 10, 500, true);

	cv::waitKey(0);




#endif

	return 0;
}
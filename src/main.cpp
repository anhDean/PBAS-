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
#include <iomanip>
#include "MotionDetector.h"

#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <stdio.h>
#include <Windows.h>
#include <iostream>

#include<opencv2\highgui\highgui.hpp>

#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__<< "(" << __LINE__ << ")" << (x) << std::endl
#else
#define LOG_MESSAGE(x)
#endif

//#define _DATASET
//#define _FOLDER
//#define _CAMERA
#define _PROTOTYPING1
//#define _PROTOTYPING2

int main(int argc, char** argv)
{
	// PBAS Parameter Settings using macro trick
#define DEFINE_FRAMEPROCESSORPARAMS(type, name, value) type name = value;
	#include "FrameProcessorParams.h"
	#undef DEFINE_FRAMEPROCESSORPARAMS

#ifdef _DATASET
	CV_Assert(argc == 3);
	std::string data_root = argv[1], result_root = argv[2];
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
	int divisor = floor(f_obj_arr.at(0).getFolderCount() / THREAD_NUM);

	std::vector<std::string> inputFolders = f_obj_arr.at(0).getInputFolders();
	std::vector<std::string> outputFolders = f_obj_arr.at(0).getOutputFolders();
	std::thread thread_arr[THREAD_NUM];
	
	int t_init = cv::getTickCount();
	for (int it = 0; it < divisor; ++it)
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

	const std::string inputFolder = "E:\\Datasets\\datasets2012\\dataset\\intermittentObjectMotion\\streetLight\\input";
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

#ifdef _PROTOTYPING1

		cv::Mat img, result;

		PBASFrameProcessor *processor = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
	
		std::string inputFolder = "E:\\Datasets\\datasets2014\\dataset\\baseline\\pedestrians\\input";
		int offset = 0;
		for (int i = 0; i < 3000 - 1; ++i)
		{	
			std::stringstream ss1, ss2;

			ss1 << inputFolder << "\\in" << std::setw(6) << std::setfill('0') << i + 1 + offset << std::setfill(' ') << ".jpg";
			img = cv::imread(ss1.str());


			processor->process(img, result);
			cv::imshow("result", result);
			cv::imshow("raw result", processor->getRawOutput());
			cv::imshow("input", img);
			cv::imshow("noise map", processor->getNoiseMap());
			cv::imshow("bg sample", processor->drawBgSample());
			if(processor->getOutputWithBb().data != NULL)
				cv::imshow("out bb", processor->getOutputWithBb());

			cv::waitKey(1);

		}



#endif

		using namespace cv;
#ifdef _PROTOTYPING2

		FrameProcessor *processor = new PBASFrameProcessor(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, subsamplingLowerBound, subsamplingUpperBound);
		std::string input = "E:\\Datasets\\datasets2012\\dataset\\baseline\\highway\\input\\in000090.jpg";
		Mat img = imread(input);
		Mat src_gray, result;
		cvtColor(img, src_gray, CV_BGR2GRAY);


		cv::Mat sobelX, sobelY, inputGray, gradMagnMap;

		cv::GaussianBlur(img, inputGray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		cv::cvtColor(inputGray, inputGray, CV_BGR2GRAY);
		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel(inputGray, sobelX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(inputGray, sobelY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
		convertScaleAbs(sobelX, sobelX);
		convertScaleAbs(sobelY, sobelY);


		imshow("sobel x", sobelX);
		imshow("sobel y", sobelY);
		waitKey(0);


#endif




return 0;
}
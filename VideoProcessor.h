#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include <iostream>
#include <iomanip>

class FrameProcessor;

class VideoProcessor
{
public:
	VideoProcessor() : callIt(true), delay(0),
		fnumber(0), stop(false), frameToStop(-1) {}

	~VideoProcessor(void);
	
	// set the callback function that
	// will be called for each frame
	void setFrameProcessor(void (*frameProcessingCallback)(cv::Mat&, cv::Mat&));
	void setFrameProcessor(FrameProcessor* frameProcessorPtr);

	// set the name of the video file
	bool setInput(std::string);
	bool setInput(const std::vector<std::string>& imgs);

	// to display the processed frames
	void displayInput(std::string);
	// to display the processed frames
	void displayOutput(std::string) ;
	// do not display the processed frames
	void dontDisplay();

	// to grab (and process) the frames of the sequence
	void run();

	// Stop the processing
	void stopIt();

	// Is the process stopped?
	bool isStopped();

	// Is a capture device opened?
	bool isOpened();

	double getFrameRate();

	// set a delay between each frame
	// 0 means wait at each frame
	// negative means no delay
	void setDelay(int);

	// return the frame number of the next frame
	long getFrameNumber();
	long getTotalNumberOfFrames();

	// the callback function to be called
	// for the processing of each frame
	void (*process)(cv::Mat&, cv::Mat&);
	FrameProcessor* frameProcessor;
	void stopAtFrameNo(long);

	bool setOutput(std::string, int, double, bool);
	bool setOutput(std::string, const std::string&, int,int);
	int getCodec(char[4]);

private:
	// The Mixture of Gaussian object
	// used with all default parameters
	cv::BackgroundSubtractorMOG mog;
	cv::BackgroundSubtractorMOG2 mog2;

	// foreground binary image
	cv::Mat foreground, objectImage;
	std::vector<std::vector<cv::Point>> contours;

	std::vector<cv::Point> massCenter;
	
	//own objects
	//ObjectInteraction oi;

	// to get the next frame
	// could be: video file or camera
	bool readNextFrame(cv::Mat&);

	// process callback to be called
	void callProcess();

	// do not call process callback
	void dontCallProcess();


	
	// the OpenCV video capture object
	cv::VideoCapture capture;

	// a bool to determine if the
	// process callback will be called
	bool callIt;
	// Input display window name
	std::string windowNameInput;
	// Output display window name
	std::string windowNameOutput;
	// delay between each frame processing
	int delay;
	// number of processed frames
	long fnumber;
	// stop at this frame number
	long frameToStop;
	// to stop the processing
	bool stop;
/*###################################################################################################################*/
/*Image Sequence*/
/*###################################################################################################################*/
	// vector of image filename to be used as input
	std::vector<std::string> images;
	// image vector iterator
	std::vector<std::string>::const_iterator itImg;

/*###################################################################################################################*/
	/*WRITE VIDEO*/
/*###################################################################################################################*/
	// the OpenCV video writer object
	cv::VideoWriter writer;
	// output filename
	std::string outputFile;
	// current index for output images
	int currentIndex;
	// number of digits in output image filename
	int digits;
	// extension of output images
	std::string extension;
	void writeNextFrame(cv::Mat&);

	cv::Size getFrameSize();
	cv::Size frameSize;
};

#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include <iostream>
#include <iomanip>

class VideoProcessor
{
private:
	// a bool to determine if the
	// process callback will be called
	bool m_callIt, m_stop;
	// Input display window name
	mutable std::string m_windowNameInput,m_windowNameOutput;
	// Output display window name
	//WRITE VIDEO

	cv::VideoWriter m_writer; // the OpenCV video writer object
	std::string m_outputFile; // output filename
	std::string m_extension; // extension of output images
	int m_currentIndex; // current index for output images
	int m_digits; // number of digits in output image filename

	// delay between each frame processing
	int m_delay;
	// number of processed frames
	long m_fnumber, m_frameToStop;
	// the OpenCV video capture object
	cv::VideoCapture m_capture;
	// foreground binary image
	cv::Mat m_foreground, m_objectImage;
	cv::Size m_frameSize;
	
	std::vector<std::string> m_images; // vector of image filename to be used as input
	std::vector<std::string>::const_iterator m_itImg; // image vector iterator


	// member functions
	bool readNextFrame(cv::Mat&); // could be: video file or camera
	// process callback to be called

	// stop at this frame number
	// to stop the processing
	void writeNextFrame(cv::Mat&);
	const cv::Size & getFrameSize() const;

	

public:

	VideoProcessor(const std::string &filename);
	VideoProcessor::VideoProcessor(const std::vector<std::string>& imgs);
	~VideoProcessor(void);
	// set a delay between each frame
	// 0 means wait at each frame
	// negative means no delay
	void setDelay(int);
	int getCodec(char[4]);
	void(*process)(cv::Mat&, cv::Mat&);
	FrameProcessor* m_frameProcessor;
	// the callback function to be called
	// for the processing of each frame
	void stopAtFrameNo(long);

	// set the callback function that
	// will be called for each frame
	

	// setters
	// set the name of the video file
	//bool init(const std::string&);
	//bool init(const std::vector<std::string>& imgs);
	void setFrameProcessor(void(*frameProcessingCallback)(cv::Mat&, cv::Mat&));
	void setFrameProcessor(FrameProcessor* frameProcessorPtr);
	bool setOutput(std::string filename, int, double, bool);
	bool setOutput(std::string filename, const std::string& ext, int numberOfDigits = 3, int startIndex = 0);
	void setStop(); // Stop the processing
	void setCallProcess();
	void unsetCallProcess(); // do not call process callback
	// getters
	double getFrameRate() const; 	
	long getFrameNumber() const; // return the frame number of the next frame
	long getTotalNumberOfFrames() const;

	// logical query
	bool isStopped(); // Is the process stopped?
	bool isOpened(); // Is a capture device opened?
	
	// display
	void displayInput(const std::string& windowName) const;// to display the processed frames
	void displayOutput(const std::string& windowName) const; // to display the processed frames
	void stopDisplay() const;  // do not display the processed frames

	void run(); // to grab (and process) the frames of the sequence

};

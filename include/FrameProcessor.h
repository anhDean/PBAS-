#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

class FrameProcessor
{

public:
	FrameProcessor(){};
	virtual ~FrameProcessor(){};	
	// processing method
	virtual void process(cv:: Mat &input, cv:: Mat &output)= 0;	
	virtual void process(cv:: Mat &input)= 0;
	virtual void resetProcessor() = 0;
	virtual void updateNoiseMap() = 0;
	virtual std::auto_ptr<cv::Mat> getBackgroundDynamics() const = 0;
	virtual const cv::Mat& getNoiseMap() const = 0;
	virtual const cv::Mat& getGradMagnMap() const = 0;
};

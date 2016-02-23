#pragma once
#include <opencv2/opencv.hpp>

class FrameProcessor
{
public:
	FrameProcessor(void){};
	~FrameProcessor(void){};
	
	// processing method
	virtual void process(cv:: Mat &input, cv:: Mat &output)= 0;	
	virtual void process(cv:: Mat &input)= 0;	
	
};

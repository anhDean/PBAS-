#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include "PBAS.h"
#include <thread>
#include <memory>
#include "PBASFeature.h"
#include "MotionDetector.h"

class PBASFrameProcessor : public FrameProcessor  

{

private:
	PBAS<PBASFeature> m_pbas;
	MotionDetector m_motionDetector;
	cv::Mat m_outputWithBb;
	int m_iteration;
	//void parallelBackgroundAveraging(std::vector<cv::Mat>* rgb, bool wGC, cv::Mat * pbasR) const;
	bool modelInitialized = false;

public:
	PBASFrameProcessor(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound); // double, int);//const for graphCuts
	virtual ~PBASFrameProcessor(void);
	
	void setDefaultValues(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate,
		double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound);
	
	
	const cv::Mat& getOutputWithBb() const;
	virtual void resetProcessor();
	virtual void process(cv::Mat &, cv::Mat &);
	virtual void process(cv::Mat &);
	virtual const cv::Mat& getRawOutput() const;
	virtual const cv::Mat getBackgroundDynamics() const;
	virtual const cv::Mat drawBgSample();
	
};

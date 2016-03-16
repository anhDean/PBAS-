#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include "PBAS.h"
#include <thread>
#include <memory>

class PBASFrameProcessor : public FrameProcessor  

{

private:
	PBAS *m_pbas1, *m_pbas2, *m_pbas3;
	int m_iteration;
	void parallelBackgroundAveraging(std::vector<cv::Mat>* rgb, bool wGC, cv::Mat * pbasR) const;
	

public:
	PBASFrameProcessor(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound); // double, int);//const for graphCuts
	virtual ~PBASFrameProcessor(void);
	
	void setDefaultValues(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate,
		double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound);
	
	virtual void resetProcessor();
	virtual void process(cv::Mat &, cv::Mat &);
	virtual void process(cv::Mat &);
	virtual std::auto_ptr<cv::Mat> getBackgroundDynamics() const;
	
};

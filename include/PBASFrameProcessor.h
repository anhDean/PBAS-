#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include "PBAS.h"
#include <thread>

class PBASFrameProcessor : public FrameProcessor  

{

private:
	cv::Mat m_lastResult, m_lastResultPP, m_currentResult, m_currentResultPP, m_noiseMap;
	PBAS *m_pbas1, *m_pbas2, *m_pbas3;
	int m_iteration;
	void parallelBackgroundAveraging(std::vector<cv::Mat>* rgb, bool wGC, cv::Mat * pbasR) const;
	void updateNoiseMap();
	

public:
	PBASFrameProcessor(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound); // double, int);//const for graphCuts
	~PBASFrameProcessor(void);
	
	void setDefaultValues(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate,
		double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound);
	
	void resetProcessor();
	void process(cv::Mat &, cv::Mat &);
	void process(cv::Mat &);
	const cv::Mat& getBackgroundDynamics() const;
	const cv::Mat& getNoiseMap() const;
	
};

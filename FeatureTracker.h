#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include "VideoProcessor.h"
#include "PBAS.h"
#include <thread>

class FeatureTracker : public FrameProcessor  

{
public:
	FeatureTracker( double resPara,
		int newN, double newR, int newRaute, int newTemporal,
		double rT, double rID, double iTS, double dTS, int dTRS, int iTRS, //const for pbas
		double newA, double newB); // double, int);//const for graphCuts
	~FeatureTracker(void);
	void process(cv::Mat &, cv::Mat &);
	void process(cv::Mat &);

private:
	cv::Mat m_pbasResult;
	PBAS m_pbas1, m_pbas2, m_pbas3;
	void parallelBackgroundAveraging(PBAS* m_pbas1, PBAS* m_pbas2, PBAS* m_pbas3, std::vector<cv::Mat>* rgb, bool wGC, cv::Mat * pbasR) const;	

};

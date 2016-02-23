#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include "VideoProcessor.h"
#include "PBAS.h"
#include <ppl.h>
#include <windows.h>


class FeatureTracker : public FrameProcessor  

{
public:
	FeatureTracker(VideoProcessor*, double, 
		int, double , int , int, 
		double, double, double, double, int, int, //const for pbas
		double, double, double, double); // double, int);//const for graphCuts
	~FeatureTracker(void);
	void process(cv::Mat &, cv::Mat &);
	void process(cv::Mat &);

private:
	void parallelBackgroundAveraging(PBAS* m_pbas1, PBAS* m_pbas2, PBAS* m_pbas3, std::vector<cv::Mat>* rgb, bool wGC, cv::Mat * pbasR);
	VideoProcessor* vp;

	//pbas segementor
	PBAS m_pbas1, m_pbas2, m_pbas3;


	int m_resizeParam;
	cv::Mat m_pbasResult;
	
	Concurrency::task_group m_tg;

};

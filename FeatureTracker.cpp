#include "FeatureTracker.h"
#include "LBSP.h"



FeatureTracker::FeatureTracker(double resPara,
							   int newN, double newR, int newRaute, int newTemporal, //const for pbas
							   double rT, double rID, double iTS, double dTS, int dTRS, int iTRS, //const for pbas
							   double newA, double newB) // double newLabelThresh, int newNeighbour) //const for graphCuts

{
	//create a pbas-regulator for every rgb channel
	m_pbas1.initialization(newN,newR,newRaute,newTemporal,newA,newB,  rT, rID, iTS, dTS, dTRS,iTRS);
	m_pbas2.initialization(newN,newR,newRaute,newTemporal,newA,newB,  rT, rID, iTS, dTS, dTRS,iTRS);
	m_pbas3.initialization(newN,newR,newRaute,newTemporal,newA,newB,  rT, rID, iTS, dTS, dTRS,iTRS);
}

FeatureTracker::~FeatureTracker(void)
{ 	
}


void FeatureTracker::process(cv:: Mat &frame, cv:: Mat &output) 
{
		//###################################
		//PRE-PROCESSING
		//check if bluring is necessary or beneficial at this point
		cv::Mat blurImage;
		cv::GaussianBlur(frame, blurImage, cv::Size(3,3), 3);
		//maybe use a bilateralFilter
		//cv::bilateralFilter(frame, blurImage, 5, 15, 15);
		//###################################
		//color image
		std::vector<cv::Mat> rgbChannels(3);
		cv::split(blurImage, rgbChannels);
		parallelBackgroundAveraging(&m_pbas1,&m_pbas2,&m_pbas3, &rgbChannels, false, &m_pbasResult);
					
		rgbChannels.at(0).release();
		rgbChannels.at(1).release();
		rgbChannels.at(2).release();
		rgbChannels.clear();
		//###############################################
		//POST-PROCESSING HERE
		//for the final results in the changedetection-challenge a 9x9 median filter has been applied
		cv::medianBlur(m_pbasResult, m_pbasResult, 9);
		//###############################################
		m_pbasResult.copyTo(output);
		//blurImage.release();

}

void FeatureTracker::parallelBackgroundAveraging(PBAS* m_pbas1, PBAS* m_pbas2, PBAS* m_pbas3, std::vector<cv::Mat>* rgb,  bool wGC,cv::Mat * pbasR) const
{	
	cv::Mat pbasResult1, pbasResult2, pbasResult3;

	std::thread t1(&PBAS::process, m_pbas1, &rgb->at(0), &pbasResult1);
	std::thread t2(&PBAS::process, m_pbas2, &rgb->at(1), &pbasResult2);
	std::thread t3(&PBAS::process, m_pbas3, &rgb->at(2), &pbasResult3);

	if (t1.joinable())
		t1.join();
	if (t2.joinable())
		t2.join();
	if (t3.joinable())
		t3.join();

	//just or all foreground results of each rgb channel
	cv::bitwise_or(pbasResult1, pbasResult3, pbasResult1);
	cv::bitwise_or(pbasResult1, pbasResult2, pbasResult1);
	pbasResult1.copyTo(*pbasR);
	
	pbasResult2.release();
	pbasResult3.release();
	pbasResult1.release();

	t1.~thread();
	t2.~thread();
	t3.~thread();

}

void FeatureTracker::process(cv::Mat &)
{
}

#include "StdAfx.h"
#include "FeatureTracker.h"



FeatureTracker::FeatureTracker(VideoProcessor *vp, double resPara,
							   int newN, double newR, int newRaute, int newTemporal, //const for pbas
							   double rT, double rID, double iTS, double dTS, int dTRS, int iTRS, //const for pbas
							   double newA, double newB, double newCF, double newCB) // double newLabelThresh, int newNeighbour) //const for graphCuts

{
	this->vp = vp;
	
	//create a pbas-regulator for every rgb channel
	m_pbas1.initialization(newN,newR,newRaute,newTemporal,newA,newB, newCF, newCB, rT, rID, iTS, dTS, dTRS,iTRS);
	m_pbas2.initialization(newN,newR,newRaute,newTemporal,newA,newB, newCF, newCB, rT, rID, iTS, dTS, dTRS,iTRS);
	m_pbas3.initialization(newN,newR,newRaute,newTemporal,newA,newB, newCF, newCB, rT, rID, iTS, dTS, dTRS,iTRS);

	m_resizeParam = resPara;

	//shadow detection

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
		

		//###############################################
		m_pbasResult.copyTo(output);
		//blurImage.release();

}

void FeatureTracker::parallelBackgroundAveraging(PBAS* m_pbas1, PBAS* m_pbas2, PBAS* m_pbas3, std::vector<cv::Mat>* rgb,  bool wGC,cv::Mat * pbasR)
{	
	cv::Mat pbasResult1, pbasResult2, pbasResult3;

	//use parallel computing for speed improvements, but higher cpu-workload
	m_tg.run([m_pbas1, rgb, &pbasResult1](){
		m_pbas1->process(&rgb->at(0), &pbasResult1);
	});

	m_tg.run([m_pbas2, rgb, &pbasResult2](){
			m_pbas2->process(&rgb->at(1), &pbasResult2);
	});

	m_tg.run_and_wait([m_pbas3, rgb, &pbasResult3](){
			m_pbas3->process(&rgb->at(2), &pbasResult3);
	});
	

	//just or all foreground results of each rgb channel
	cv::bitwise_or(pbasResult1, pbasResult3, pbasResult1);
	cv::bitwise_or(pbasResult1, pbasResult2, pbasResult1);
	cv::medianBlur(pbasResult1, pbasResult1, 9);
	pbasResult1.copyTo(*pbasR);
	
	pbasResult2.release();
	pbasResult3.release();
	pbasResult1.release();
}

void FeatureTracker::process(cv::Mat &)
{
}

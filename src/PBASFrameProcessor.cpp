#include "PBASFrameProcessor.h"
#include "LBSP.h"

PBASFrameProcessor::PBASFrameProcessor(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, 
	double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound) : 
	m_pbas1(new PBAS()), m_pbas2(new PBAS()), m_pbas3(new PBAS())// double newLabelThresh, int newNeighbour) //const for graphCuts

{
	setDefaultValues(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, samplingLowerBound, samplingUpperBound);
}


PBASFrameProcessor::~PBASFrameProcessor(void)
{ 	
	delete m_pbas1;
	delete m_pbas2;
	delete m_pbas3;
}


void PBASFrameProcessor::setDefaultValues(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate,
	double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound)
{
	m_pbas1->initialization(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, samplingLowerBound, samplingUpperBound);
	m_pbas2->initialization(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, samplingLowerBound, samplingUpperBound);
	m_pbas3->initialization(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, samplingLowerBound, samplingUpperBound);
}


void PBASFrameProcessor::process(cv:: Mat &frame, cv:: Mat &output)
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
		parallelBackgroundAveraging(&rgbChannels, false, &m_pbasResult);
					
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

void PBASFrameProcessor::parallelBackgroundAveraging(std::vector<cv::Mat>* rgb,  bool wGC,cv::Mat * pbasR) const
{	
	cv::Mat pbasResult1, pbasResult2, pbasResult3;

	std::thread t1(&PBAS::process, m_pbas1, &rgb->at(0), &pbasResult1);
	std::thread t2(&PBAS::process, m_pbas2, &rgb->at(1), &pbasResult2);
	std::thread t3(&PBAS::process, m_pbas3, &rgb->at(2), &pbasResult3);

	if (t1.joinable() && t2.joinable() && t3.joinable())
	{
		t1.join(); t2.join(); t3.join();
	}


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

void PBASFrameProcessor::process(cv::Mat &)
{
}


void PBASFrameProcessor::resetProcessor()
{
	// performs reset on PBAS members: background model is deleted etc.
	m_pbas1->reset();
	m_pbas2->reset();
	m_pbas3->reset();
}
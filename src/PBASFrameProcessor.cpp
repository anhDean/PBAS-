#include "PBASFrameProcessor.h"


PBASFrameProcessor::PBASFrameProcessor(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, 
	double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound) : 
	m_iteration(0), m_pbas(PBAS<PBASFeature>()) // double newLabelThresh, int newNeighbour) //const for graphCuts

{
	setDefaultValues(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, samplingLowerBound, samplingUpperBound);
}
PBASFrameProcessor::~PBASFrameProcessor(void)
{ 	
}

void PBASFrameProcessor::setDefaultValues(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate,
	double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound)
{
	m_pbas.initialization(N, defaultR, minHits, defaultSubsampling, alpha, beta, RScale, RIncDec, subsamplingIncRate, subsamplingDecRate, samplingLowerBound, samplingUpperBound);

}

void PBASFrameProcessor::process(cv:: Mat &frame, cv:: Mat &output)
{
		const int medFilterSize = 9;
		double meanGradMagn;

		if (m_iteration == 0)
		{
			m_lastResult = cv::Mat::zeros(frame.size(), CV_8U);
			m_lastResultPP = cv::Mat::zeros(frame.size(), CV_8U);
			m_noiseMap = cv::Mat::zeros(frame.size(), CV_32F);
			m_gradMagnMap.create(frame.size(), CV_8U);
		}

		updateGradMagnMap(frame);
		std::vector<PBASFeature> featureMap = PBASFeature::calcFeatureMap(frame, m_gradMagnMap);
		meanGradMagn = PBASFeature::calcMeanGradMagn(featureMap, frame.rows, frame.cols);
		PBASFeature::setColorWeight(0.8 - meanGradMagn / 255);
		
		std::cout << "Mean gradient magnitude in percentage: "<< 100 * meanGradMagn / 255 << "%"<<  std::endl;
		
		m_pbas.process(featureMap, frame.rows, frame.cols, m_currentResult);
		//parallelBackgroundAveraging(&rgbChannels, false, &m_currentResult);
		//###############################################
		//POST-PROCESSING HERE
		//for the final results in the changedetection-challenge a 9x9 median filter has been applied
		cv::medianBlur(m_currentResult, m_currentResultPP, medFilterSize);

		//###############################################
		m_currentResultPP.copyTo(output);
		updateNoiseMap();
		m_currentResult.copyTo(m_lastResult);
		++m_iteration;
}
/*
void PBASFrameProcessor::parallelBackgroundAveraging(std::vector<cv::Mat>* rgb,  bool wGC,cv::Mat * pbasR) const
{	
	cv::Mat pbasResult1, pbasResult2, pbasResult3;

	std::thread t1(&PBAS::process, m_pbas1, &rgb->at(0), &pbasResult1, m_gradMagnMap, m_noiseMap);
	std::thread t2(&PBAS::process, m_pbas2, &rgb->at(1), &pbasResult2, m_gradMagnMap, m_noiseMap);
	std::thread t3(&PBAS::process, m_pbas3, &rgb->at(2), &pbasResult3, m_gradMagnMap, m_noiseMap);

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
*/
void PBASFrameProcessor::process(cv::Mat &)
{
}
void PBASFrameProcessor::resetProcessor()
{
	// performs reset on PBAS members: background model is deleted etc.
	m_iteration = 0;
	m_lastResult.release();
	m_lastResultPP.release();
	m_noiseMap.release();
	m_gradMagnMap.release();
	m_pbas.reset();

}
std::auto_ptr<cv::Mat> PBASFrameProcessor::getBackgroundDynamics() const
{
	std::auto_ptr<cv::Mat> bgdyn(new cv::Mat());
	*bgdyn = m_pbas.getSumMinDistMap().clone();
	*bgdyn = bgdyn->mul((float)1.0 / m_pbas.getRuns());
	FrameProcessor::normalizeMat(*bgdyn);
	return bgdyn;
}


const cv::Mat& PBASFrameProcessor::getRawOutput() const
{
	return m_currentResult;
}

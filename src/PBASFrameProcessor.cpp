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
		
		//std::cout << "Mean gradient magnitude in percentage: "<< 100 * meanGradMagn / 255 << "%"<<  std::endl;
		m_pbas.process(featureMap, frame.rows, frame.cols, m_currentResult);
		// normalize gradient magnmap

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
const cv::Mat PBASFrameProcessor::getBackgroundDynamics() const
{
	cv::Mat bgdyn = m_pbas.getSumMinDistMap() / (m_pbas.getRuns() + 1);
	FrameProcessor::normalizeMat(bgdyn);
	return bgdyn;
}


const cv::Mat& PBASFrameProcessor::getRawOutput() const
{
	return m_currentResult;
}


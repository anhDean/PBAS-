#include "MoGFrameProcessor.h"
#include "ColourFeature.h"
#include "LBSP.h"


MoGFrameProcessor::MoGFrameProcessor(int K, int dim, float decisionThresh, float alpha, float rho) : m_processor(new MoGSegmenter<ColourFeature>(K, dim, decisionThresh, alpha, rho)), m_K(K), m_dim(dim)
{
}
void MoGFrameProcessor::process(cv::Mat &frame, cv::Mat &output)
{
	const int medFilterSize = 9;
	//###################################
	//PRE-PROCESSING
	//check if bluring is necessary or beneficial at this point
	if (m_iteration == 0)
	{
		m_lastResult = cv::Mat::zeros(frame.size(), CV_8U);
		m_lastResultPP = cv::Mat::zeros(frame.size(), CV_8U);
		m_noiseMap = cv::Mat::zeros(frame.size(), CV_64F);
		m_gradMagnMap.create(frame.size(), CV_32F);
		m_processor->init(frame.rows, frame.cols);
	}
	updateGradMagnMap(frame);
	std::vector<ColourFeature> featureMap = ColourFeature::calcFeatureMap(frame);
	//maybe use a bilateralFilter
	//cv::bilateralFilter(frame, blurImage, 5, 15, 15);
	//###################################
	//color image
	cv::Mat result;
	m_processor->processFrame(featureMap, result);
	//###############################################
	//POST-PROCESSING HERE
	//for the final results in the changedetection-challenge a 9x9 median filter has been applied
	result.copyTo(m_currentResult);
	cv::medianBlur(m_currentResult, m_currentResultPP, medFilterSize);
	//###############################################
	m_currentResultPP.copyTo(output);
	//m_currentResult.copyTo(output);
	updateNoiseMap();
	m_currentResult.copyTo(m_lastResult);
	++m_iteration;
}

void MoGFrameProcessor::resetProcessor()
{
	m_processor = std::auto_ptr<MoGSegmenter<ColourFeature>>(new MoGSegmenter<ColourFeature>(m_K, m_dim));
}


MoGFrameProcessor::~MoGFrameProcessor()
{

}

std::auto_ptr<cv::Mat> MoGFrameProcessor::getBackgroundDynamics() const
{
	return m_processor->getVarianceTraceMat();
}
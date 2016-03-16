#include "MoGFrameProcessor.h"
#include "ColourFeature.h"
#include "LBSP.h"


MoGFrameProcessor::MoGFrameProcessor(int K, int dim, float decisionThresh, float alpha, float rho) : processor(K, dim, decisionThresh, alpha, rho)
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
		processor.init(frame.rows, frame.cols);
	}
	std::vector<ColourFeature> featureMap = ColourFeature::calcFeatureMap(frame);
	//maybe use a bilateralFilter
	//cv::bilateralFilter(frame, blurImage, 5, 15, 15);
	//###################################
	//color image
	cv::Mat result;
	processor.processFrame(featureMap, result);
	//###############################################
	//POST-PROCESSING HERE
	//for the final results in the changedetection-challenge a 9x9 median filter has been applied
	result.copyTo(m_currentResult);
	cv::medianBlur(m_currentResult, m_currentResultPP, medFilterSize);
	//###############################################
	//m_currentResultPP.copyTo(output);
	m_currentResult.copyTo(output);
	updateNoiseMap();
	m_currentResult.copyTo(m_lastResult);
	++m_iteration;
}

void resetProcessor()
{

}


MoGFrameProcessor::~MoGFrameProcessor()
{

}
void MoGFrameProcessor::resetProcessor()
{

}

std::auto_ptr<cv::Mat> MoGFrameProcessor::getBackgroundDynamics() const
{
	//TODO: implement 
	std::auto_ptr<cv::Mat> x;
	return x;
}
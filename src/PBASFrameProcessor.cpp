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
		std::vector<AmbiguousCandidate> candidates;
		cv::Mat subSamplingOffset = cv::Mat::zeros(frame.size(), CV_32F);

		if (m_iteration == 0)
		{
			m_lastResult = cv::Mat::zeros(frame.size(), CV_8U);
			m_lastResultPP = cv::Mat::zeros(frame.size(), CV_8U);
			m_noiseMap = cv::Mat::zeros(frame.size(), CV_32F);
			m_gradMagnMap.create(frame.size(), CV_8U);
			m_motionDetector.initialize(frame);
		}
		if (m_iteration != 0)
			m_motionDetector.updateMotionMap(frame);

		updateGradMagnMap(frame);
		std::vector<PBASFeature> featureMap = PBASFeature::calcFeatureMap(frame, m_gradMagnMap);
		meanGradMagn = PBASFeature::calcMeanGradMagn(featureMap, frame.rows, frame.cols);
		PBASFeature::setColorWeight(0.8 - meanGradMagn / 255);
		
		m_pbas.process(featureMap, frame.rows, frame.cols, m_currentResult, m_motionDetector.getMotionMap());

		if (m_pbas.isInitialized())
		{
			m_motionDetector.updateCandidates(m_currentResultPP);
			m_motionDetector.classifyStaticCandidates(frame, m_pbas.drawBgSample(), m_currentResultPP);

			candidates = m_motionDetector.getStaticObjects();

			for (int k = 0; k < candidates.size(); ++k)
			{
				if (candidates[k].state == DETECTOR_GHOST_OBJECT)
				{
					subSamplingOffset(candidates[k].boundingBox) -= 5;
				}
				else if ((candidates[k].state == DETECTOR_STATIC_OBJECT))
				{
					subSamplingOffset(candidates[k].boundingBox) += 5;
				}
				
			}
			m_pbas.subSamplingOffset(subSamplingOffset);
			
			#ifndef _DATASET	
			MotionDetector::drawBoundingBoxesClassified(m_currentResultPP, m_outputWithBb, candidates);
			#endif 
		}



// !_DATASET

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

const cv::Mat PBASFrameProcessor::drawBgSample()
{
	cv::Mat bgdyn = m_pbas.drawBgSample();
	return bgdyn;
}


const cv::Mat& PBASFrameProcessor::getOutputWithBb() const
{
	return m_outputWithBb;
}
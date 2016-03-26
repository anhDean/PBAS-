#pragma once
#include "FrameProcessor.h"
#include "LBSP.h"
#include "MoGSegmenter.h"
#include "BackgroundFeature.h"
#include "PBASFeature.h"

class MoGFrameProcessor : public FrameProcessor
{
private:
	int m_K, m_dim;
	cv::Mat m_gradMagnMap;
	std::auto_ptr<MoGSegmenter<PBASFeature>> m_processor;
public:
	MoGFrameProcessor(int K, int dim, float decisionThresh = 2.5, float alpha = 0.075, float rho = 0.01);
	virtual ~MoGFrameProcessor();
	virtual void resetProcessor();
	virtual void process(cv::Mat &in, cv::Mat &out);
	virtual void process(cv::Mat &in) {}
	virtual const cv::Mat getBackgroundDynamics() const;
	virtual const cv::Mat& getRawOutput() const;
};


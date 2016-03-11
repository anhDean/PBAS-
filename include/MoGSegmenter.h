#pragma once
#include <opencv2\opencv.hpp>


template <typename MoGFeature>
class MoGSegmenter
{

private:
	const int m_K, m_width, m_height, m_nFeatures;
	float m_decisionThreshScale, m_learnRate;

	static const int FOREGROUNDVAL, BACKGROUNDVAL;

	std::vector<cv::Mat> m_stdDevMat, m_weights;
	std::vector<std::vector<MoGFeature> > m_meanMat, m_meanDiff;  // outer vector holds k_i Mixtures, inner vector spans matrix with MoGFeature class



public:
	MoGSegmenter(int K, int width, int height, int dim, float decisionThresh = 2.5, float learnRate = 0.01);
	~MoGSegmenter();

	int getWidth() const;
	int getHeight() const;
	int getDimensions() const;
	float getDecisionThreshScale() const;
	float getLearnRate() const;

	const std::vector<std::vector<MoGFeature> >& getMeanMat() const;
	const cv::Mat& getStdDevMat() const;
	const cv::Mat& getWeightMat() const;


	void setDecisionThreshScale(float newVal);
	void setLearnRate(float newVal);
	void processFrame(const cv::Mat& input, cv::Mat& output);

};

template <typename MoGFeature>
const int MoGSegmenter<MoGFeature>::FOREGROUNDVAL = 255;
template <typename MoGFeature>
const int MoGSegmenter<MoGFeature>::BACKGROUNDVAL = 0;


template <typename MoGFeature>
MoGSegmenter<MoGFeature>::MoGSegmenter(int K, int width, int height, int nFeatures, float decisionThresh, float learnRate) : m_K(K), m_width(width), m_height(height), m_nFeatures(nFeatures),
m_decisionThreshScale(decisionThresh), m_learnRate(learnRate), m_weights(cv::Mat())
{

}

template <typename MoGFeature>
MoGSegmenter<MoGFeature>::~MoGSegmenter()
{
}

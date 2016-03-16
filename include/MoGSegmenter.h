#pragma once
#include <opencv2\opencv.hpp>


template <typename MoGFeature>
class MoGSegmenter
{

private:
	const int m_K, m_nFeatures;
	int m_width, m_height;
	float m_decisionThreshScale, m_alpha, m_rho, m_defaultVar, m_T = 0.75;

	static const int FOREGROUNDVAL, BACKGROUNDVAL;

	std::vector<std::vector<float>> m_varianceTensor, m_weightTensor; // outer matrix K -> channel -> matrix holding values // outer vecotr K ->  matrix holding values
	std::vector<std::vector<MoGFeature>> m_meanTensor;  // outer vector holds k_i Mixtures, inner vector spans matrix with MoGFeature class


public:
	MoGSegmenter(int K, int dim, float decisionThresh = 0.7, float alpha = 0.01, float rho = 0.01);
	~MoGSegmenter();

	int getWidth() const;
	int getHeight() const;
	int getNMixtures() const;
	int getNFeatures() const;
	float getDecisionThreshScale() const;
	float getAlpha() const;
	float getRho() const;

	const std::vector<std::vector<MoGFeature> >& getMeanMat() const;
	std::auto_ptr<cv::Mat> getVarianceTraceMat() const;
	const cv::Mat& getWeightMat() const;


	void init(int height, int width);
	void setDecisionThreshScale(float newVal);
	void setAlpha(float newVal);
	void setRho(float newVal);
	void setDefaultVar(float newVal) { m_defaultVar = newVal;}

	void processFrame(const std::vector<MoGFeature>& input, cv::Mat& output);

};

template <typename MoGFeature>
const int MoGSegmenter<MoGFeature>::FOREGROUNDVAL = 255;
template <typename MoGFeature>
const int MoGSegmenter<MoGFeature>::BACKGROUNDVAL = 0;


template <typename MoGFeature>
MoGSegmenter<MoGFeature>::MoGSegmenter(int K, int nFeatures, float decisionThresh, float alpha, float rho) 
	: m_K(K), m_nFeatures(nFeatures), m_decisionThreshScale(decisionThresh), m_alpha(alpha), m_rho(rho), m_defaultVar(36), m_varianceTensor(m_K), m_meanTensor(m_K), m_weightTensor(m_K)
{
}

template <typename MoGFeature>
void MoGSegmenter<MoGFeature>::processFrame(const std::vector<MoGFeature>& input, cv::Mat& output)
{
	int minWeightIdx;
	int B;
	float varTrace, tmp_weightSumYX, minWeight,sum_weight;
	bool match;
	int tmp_idx = 0;

	std::vector<float> absdiffnormXY;

	cv::Mat segMap(m_height, m_width, CV_8U);
	segMap.setTo(cv::Scalar(FOREGROUNDVAL));

	cv::Mat weightXY(m_K, 1, CV_32F), sortedWeightIdx;

	for (int y = 0; y < m_height; ++y)
	{
		for (int x = 0; x < m_width; ++x)
		{
			match = false;
			tmp_weightSumYX = 0.0;
			for (int k = 0; k < m_K; ++k)
			{
				varTrace = 0.0;
				for (int c = 0; c < m_nFeatures; ++c) // calculate sum over all channels of variance vector
				{
					varTrace += m_varianceTensor.at(k).at(y  * m_width * m_nFeatures + x * m_nFeatures + c);
				}

				absdiffnormXY.push_back(cv::saturate_cast<float>(MoGFeature::calcDistance(m_meanTensor.at(k).at(y * m_width + x), input.at(y * m_width + x))));

				if (absdiffnormXY.at(k) <= m_decisionThreshScale *std::sqrt(varTrace))
				{	// match
					match = true;
					// update weight and learning rate
					
					m_weightTensor.at(k).at(y * m_width + x) = cv::saturate_cast<float>((1 - m_alpha) * m_weightTensor.at(k).at(y * m_width + x)) + m_alpha;
					m_rho = cv::saturate_cast<float>(m_alpha / m_weightTensor.at(k).at(y * m_width + x)); // TODO: change according to paper
					// update means
					m_meanTensor.at(k).at(y * m_width + x) = m_meanTensor.at(k).at(y * m_width + x) *  (1 - m_rho)  +  input.at(y * m_width + x) * m_rho;
					// update variance
					for (int c = 0; c < m_nFeatures; ++c) // calculate sum over all channels of variance vector
					{
						m_varianceTensor.at(k).at(y  * m_width * m_nFeatures + x * m_nFeatures + c) = cv::saturate_cast<float>(cv::saturate_cast<float>((1 - m_rho) * m_varianceTensor.at(k).at(y  * m_width * m_nFeatures + x * m_nFeatures + c))
							+ cv::saturate_cast<float>(m_rho * std::pow((input.at(y * m_width + x)[c] - m_meanTensor.at(k).at(y * m_width + x)[c]), 2)));
					}
				}
				else
				{
					// decrease weight slightly
					m_weightTensor.at(k).at(y * m_width + x) *= (1 - m_alpha);
				}
				tmp_weightSumYX += m_weightTensor.at(k).at(y * m_width + x);
			}

			// get lowest weight
			minWeight = 10;
			for (int k = 0; k < m_K; ++k)
			{
				if (m_weightTensor.at(k).at(y * m_width + x) < minWeight)
				{
					minWeight = m_weightTensor.at(k).at(y * m_width + x);
					minWeightIdx = k;
				}
			}
			// no gaussian matches -> replace gaussian with lowest weight
			if (!match)
			{
				m_meanTensor.at(minWeightIdx).at(y * m_width + x) = input.at(y * m_width + x);
				m_weightTensor.at(minWeightIdx).at(y * m_width + x) = 0.05;
				for (int c = 0; c < m_nFeatures; ++c)
				{
					m_varianceTensor.at(minWeightIdx).at(y  * m_width * m_nFeatures + x * m_nFeatures + c) = m_defaultVar;
				}
				tmp_weightSumYX += (0.05 - minWeight);
			}

			// normalize weights
			for (int k = 0; k < m_K; ++k)
			{
				m_weightTensor.at(k).at(y * m_width + x) = cv::saturate_cast<float>(m_weightTensor.at(k).at(y * m_width + x) / tmp_weightSumYX);
			}

			// get background model
			
			for (int k = 0; k < m_K; ++k)
			{
				varTrace = 0.0;
				for (int c = 0; c < m_nFeatures; ++c)
				{
					varTrace += m_varianceTensor.at(k).at(y  * m_width * m_nFeatures + x * m_nFeatures + c);
				}
				weightXY.at<float>(k, 0) = m_weightTensor.at(k).at(y * m_width + x) / std::sqrt(varTrace);
			}



			cv::Mat tmp;
			cv::sortIdx(weightXY, tmp, cv::SORT_EVERY_COLUMN | cv::SORT_DESCENDING);
			// take the B first distributions
			sum_weight = 0.0;
			B = 0;


			while (sum_weight < m_T)
			{
				sum_weight += m_weightTensor.at(tmp.at<int>(B, 0)).at(y * m_width + x);
				++B;
			}

			for (int b = 0; b < B; ++b)
			{
				varTrace = 0.0;
				tmp_idx = int(tmp.at<int>(b, 0));
				for (int c = 0; c < m_nFeatures; ++c) // calculate sum over all channels of variance vector
				{
					varTrace += m_varianceTensor.at(tmp_idx).at(y  * m_width * m_nFeatures + x * m_nFeatures + c);
				}

				if (absdiffnormXY.at(tmp_idx) <= m_decisionThreshScale *std::sqrt(varTrace))
				{
					segMap.at<uchar>(y, x) = BACKGROUNDVAL;
					break;
				}
			}
			absdiffnormXY.clear();
		}
	}
	segMap.copyTo(output);
}


template <typename MoGFeature>
int MoGSegmenter<MoGFeature>::getWidth() const
{
	return m_width;
}
template <typename MoGFeature>
int MoGSegmenter<MoGFeature>::getHeight() const
{
	return m_height;
}
template <typename MoGFeature>
int MoGSegmenter<MoGFeature>::getNFeatures() const
{
	return m_nFeatures;
}
template <typename MoGFeature>
float MoGSegmenter<MoGFeature>::getRho() const
{
	return m_rho;
}
template <typename MoGFeature>
int MoGSegmenter<MoGFeature>::getNMixtures() const
{
	return m_K;
}
template <typename MoGFeature>
float MoGSegmenter<MoGFeature>::getDecisionThreshScale() const
{
	return m_decisionThreshScale;
}
template <typename MoGFeature>
float MoGSegmenter<MoGFeature>::getAlpha() const
{
	return m_alpha;
}
template <typename MoGFeature>
const std::vector<std::vector<MoGFeature> >& MoGSegmenter<MoGFeature>::getMeanMat() const
{
	return m_meanTensor;
}
template <typename MoGFeature>
std::auto_ptr<cv::Mat> MoGSegmenter<MoGFeature>::getVarianceTraceMat() const
{
	std::auto_ptr<cv::Mat> result_ptr(new cv::Mat(m_height, m_width, CV_32F));
	result_ptr->setTo(cv::Scalar(0.0));
	double min, max;


	for (int k = 0; k < m_K; ++k)
	{
		for (int y = 0; y < m_height; ++y)
		{
			for (int x = 0; x < m_width; ++x)
			{

				for (int c = 0; c < m_nFeatures; ++c)
				{
					result_ptr->at<float>(y, x) += cv::saturate_cast<float>(m_weightTensor.at(k).at(y * m_width + x) * m_varianceTensor.at(k).at(y * m_width * m_nFeatures + x * m_nFeatures + c));
				}
			}

		}
	}
	

	cv::minMaxIdx(*result_ptr, &min, &max);
	cv::convertScaleAbs(*result_ptr, *result_ptr, 255 / max);

	return result_ptr;
}
template <typename MoGFeature>
const cv::Mat& MoGSegmenter<MoGFeature>::getWeightMat() const
{
	return m_weights;
}
template <typename MoGFeature>
void MoGSegmenter<MoGFeature>::setDecisionThreshScale(float newVal)
{
	m_decisionThreshScale = newVal;
}
template <typename MoGFeature>
void MoGSegmenter<MoGFeature>::setAlpha(float newVal)
{
	m_alpha = newVal;
}
template <typename MoGFeature>
void MoGSegmenter<MoGFeature>::setRho(float newVal)
{
	m_rho = newVal;
}


template <typename MoGFeature>
MoGSegmenter<MoGFeature>::~MoGSegmenter()
{

}


template <typename MoGFeature>
void MoGSegmenter<MoGFeature>::init(int height, int width)
{
	m_height = height;
	m_width = width;
	
	for (int k = 0; k < m_K; ++k)
	{
		for (int y = 0; y < m_height; ++y)
		{
			for (int x = 0; x < m_width; ++x)
			{
				
				for (int c = 0; c < m_nFeatures; ++c)
				{
					m_varianceTensor.at(k).push_back(m_defaultVar);
				}
				m_weightTensor.at(k).push_back(1.0 / m_K);
				m_meanTensor.at(k).push_back(MoGFeature());
			}
		}
	}
}

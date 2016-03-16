#include "BackgroundFeature.h"


BackgroundFeature::BackgroundFeature()
{
	cv::RNG randomGenerator;
	m_color.create(3, 1, CV_8S);
	cv::randu(m_color, 0, 255);
	m_regionSignatureHW = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 16));
	m_gradMagn = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
}
BackgroundFeature::BackgroundFeature(cv::Mat& color, float gradMagn, int regionSignature) : m_gradMagn(gradMagn), m_regionSignatureHW(regionSignature)
{
	m_color.release();
	m_color = color.clone();
}
BackgroundFeature::BackgroundFeature(const BackgroundFeature& other)
{
	copyOther(other);
}

void BackgroundFeature::copyOther(const BackgroundFeature& other)
{
	m_color = other.m_color.clone();
	m_gradMagn = other.m_gradMagn;
	m_regionSignatureHW = other.m_regionSignatureHW;
}
void BackgroundFeature::clear()
{
	m_color.release();
}
BackgroundFeature& BackgroundFeature::operator= (const BackgroundFeature& other)
{
	if (this != &other)
	{
		clear();
		copyOther(other);
	}
	return *this;
}
const BackgroundFeature BackgroundFeature::operator- () const
{
	BackgroundFeature result;
	result.m_color = -m_color.clone();
	result.m_gradMagn = -m_gradMagn;
	result.m_regionSignatureHW = -m_regionSignatureHW;
	return result;
}
BackgroundFeature& BackgroundFeature::operator-= (const BackgroundFeature& other)
{
	*this += -other;
	return *this;
}
const BackgroundFeature BackgroundFeature::operator- (const BackgroundFeature& other) const
{
	BackgroundFeature result = *this;
	result -= other;
	return result;
}
const BackgroundFeature BackgroundFeature::operator* (double scale) const
{
	BackgroundFeature result = *this;
	result *= scale;
	return result;
}
BackgroundFeature& BackgroundFeature::operator*= (double scale)
{
	m_color *= scale;
	m_gradMagn *= scale;
	m_regionSignatureHW *= scale;
	return *this;
}
const BackgroundFeature BackgroundFeature::operator+ (const BackgroundFeature& other) const
{
	BackgroundFeature result = *this;
	result += other;
	return result;
}
BackgroundFeature& BackgroundFeature::operator+= (const BackgroundFeature& other)
{
	m_color += other.m_color;
	m_gradMagn += other.m_gradMagn;
	m_regionSignatureHW += other.m_regionSignatureHW;
	return *this;
}
BackgroundFeature& BackgroundFeature::operator /= (double scaleFactor)
{
	*this *= 1.0 / scaleFactor;
	return *this;
}
const BackgroundFeature BackgroundFeature::operator / (double scaleFactor) const
{
	BackgroundFeature result = *this;
	result /= scaleFactor;
	return result;
}

double BackgroundFeature::calcDistance(const BackgroundFeature& first, const BackgroundFeature& second)
{
	BackgroundFeature diff = first - second;

	double result = std::pow(diff.m_color.at<char>(0, 0), 2) + std::pow(diff.m_color.at<char>(1, 0), 2) + std::pow(diff.m_color.at<char>(2, 0), 2)
		+ std::pow(diff.m_gradMagn, 2) + std::pow(diff.m_regionSignatureHW, 2);
	return std::sqrt(result);
}
const float& BackgroundFeature::operator [] (int position) const
{
	switch (position)
	{
	case 0:
		return m_color.at<char>(0, 0);
	case 1:
		return m_color.at<char>(1, 0);
	case 2:
		return m_color.at<char>(2, 0);
	case 3:
		return m_gradMagn;
	case 4:
		return m_regionSignatureHW;
	}

}


std::vector<BackgroundFeature> BackgroundFeature::calcFeatureMap(const cv::Mat& inputFrame, const cv::Mat& gradMagnMap, const cv::Mat& signatureHWMap)
{
	// assume input frame 3 channel matrix
	std::vector<BackgroundFeature> result(inputFrame.rows * inputFrame.cols);
	float tmp_gradMagn;
	int tmp_signature_weight;
	cv::Mat tmp_color(3, 1, CV_8S);

	for (int y = 0; y < inputFrame.rows; ++y)
	{

		for (int x = 0; x < inputFrame.cols; ++x)
		{
			tmp_gradMagn = gradMagnMap.at<float>(y, x);
			tmp_signature_weight = gradMagnMap.at<uchar>(y, x);
			tmp_color.at<char>(0, 0) = inputFrame.at<uchar>(y, x);
			tmp_color.at<char>(1, 0) = inputFrame.at<uchar>(y, x + 1);
			tmp_color.at<char>(2, 0) = inputFrame.at<uchar>(y, x + 2);

			result.at(y * inputFrame.cols + x) = BackgroundFeature(tmp_color, tmp_gradMagn, tmp_signature_weight);
		}
	}
	return result;
}


BackgroundFeature::~BackgroundFeature()
{

}





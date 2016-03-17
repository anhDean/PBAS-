#include "PBASFeature.h"

float PBASFeature::fg_colorWeight = 0.8;


PBASFeature::PBASFeature()
{
	cv::RNG randomGenerator;
	m_B = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_R = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_G = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_gradMagnVal = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
}
PBASFeature::PBASFeature(int R, int G, int B, int gM, float colorWeight, float gradMagnWeight)
{
	m_R = R; m_G = G; m_B = B;
	m_gradMagnVal = gM;
}
PBASFeature::PBASFeature(const cv::Mat& color, int gradMagnVal, float colorWeight)
{
	CV_Assert(color.rows == 1 && color.cols == 1 && color.type() == CV_8UC3);
	m_B = color.ptr<uchar>(0)[0];
	m_G = color.ptr<uchar>(0)[1];
	m_R = color.ptr<uchar>(0)[2];

	m_gradMagnVal = gradMagnVal;
}
PBASFeature::PBASFeature(const PBASFeature& other)
{
	copyOther(other);
}
void PBASFeature::copyOther(const PBASFeature& other)
{
	m_B = other.m_B;
	m_R = other.m_R;
	m_G = other.m_G;
	m_gradMagnVal = other.m_gradMagnVal;
}

// calc functions
double PBASFeature::calcDistance(const PBASFeature& first, const PBASFeature& second, int Lnorm)
{
	double result;
	PBASFeature tmp = first - second;
	if (Lnorm == 1)
	{
		result  = (std::abs(tmp.m_B) + std::abs(tmp.m_G) + std::abs(tmp.m_R)) * fg_colorWeight + (1- fg_colorWeight) * std::abs(tmp.m_gradMagnVal);
	}

	else
	{
		result = std::sqrt(((std::pow(tmp.m_B,2) + std::pow(tmp.m_G,2) + std::pow(tmp.m_R,2)) *fg_colorWeight + (1 - fg_colorWeight) * std::pow(tmp.m_gradMagnVal,2)));
	}
	return result;

}
std::vector<PBASFeature> PBASFeature::calcFeatureMap(const cv::Mat& inputFrame)
{
	std::vector<PBASFeature> result;
	// get gradient magnitude map
	cv::Mat sobelX, sobelY, inputGray, gradMagnMap;

	cv::GaussianBlur(inputFrame, inputGray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	cv::cvtColor(inputGray, inputGray, CV_BGR2GRAY);
	
	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(inputGray, sobelX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(sobelX, sobelX);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(inputGray, sobelY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(sobelY, sobelY);

	/// Total Gradient (approximate)
	cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, gradMagnMap);


	cv::Mat tmp(1, 1, CV_8UC3);

	for (int y = 0; y < inputFrame.rows; ++y)
	{
		for (int x = 0; x < inputFrame.cols; ++x)
		{

			tmp.ptr<uchar>(0)[0] = inputFrame.at<cv::Vec3b>(y, x)[0];
			tmp.ptr<uchar>(0)[1] = inputFrame.at<cv::Vec3b>(y, x)[1];
			tmp.ptr<uchar>(0)[2] = inputFrame.at<cv::Vec3b>(y, x)[2];

			result.push_back(PBASFeature(tmp, gradMagnMap.at<uchar>(y,x)));
		}


	}
	return result;

}
std::vector<PBASFeature> PBASFeature::calcFeatureMap(const cv::Mat& inputFrame, const cv::Mat& gradMagnMap)
{
	std::vector<PBASFeature> result;
	cv::Mat tmp(1, 1, CV_8UC3);

	for (int y = 0; y < inputFrame.rows; ++y)
	{
		for (int x = 0; x < inputFrame.cols; ++x)
		{

			tmp.ptr<uchar>(0)[0] = inputFrame.at<cv::Vec3b>(y, x)[0];
			tmp.ptr<uchar>(0)[1] = inputFrame.at<cv::Vec3b>(y, x)[1];
			tmp.ptr<uchar>(0)[2] = inputFrame.at<cv::Vec3b>(y, x)[2];

			result.push_back(PBASFeature(tmp, gradMagnMap.at<uchar>(y, x)));
		}
	}
	return result;
}
float PBASFeature::calcMeanGradMagn(const std::vector<PBASFeature>& featureMap, int height, int width)
{
	float result = 0.0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{

			result += featureMap.at(y * width + x).m_gradMagnVal;
		}
	}
	result /= (width * height);
	return result;
}


// operator overloading
PBASFeature& PBASFeature::operator= (const PBASFeature& other)
{
	if (this != &other)
	{
		copyOther(other);
	}
	return *this;
}
const PBASFeature PBASFeature::operator- () const
{
	PBASFeature result;
	result.m_B = -m_B;
	result.m_G = -m_G;
	result.m_R = -m_R;
	result.m_gradMagnVal = -m_gradMagnVal;
	return result;
}
PBASFeature& PBASFeature::operator+= (const PBASFeature& other)
{
	m_B += other.m_B;
	m_G += other.m_G;
	m_R += other.m_R;
	m_gradMagnVal += other.m_gradMagnVal;
	return *this;
}
const PBASFeature PBASFeature::operator+ (const PBASFeature& other) const
{
	PBASFeature result = *this;
	result += other;
	return result;
}
PBASFeature& PBASFeature::operator-= (const PBASFeature& other)
{
	*this += -other;
	return *this;
}
const PBASFeature PBASFeature::operator- (const PBASFeature& other) const
{
	PBASFeature result = *this;
	result -= other;
	return result;
}
const PBASFeature PBASFeature::operator* (double scale) const
{
	PBASFeature result = *this;
	result *= scale;
	return result;
}
PBASFeature& PBASFeature::operator*= (double scale)
{
	m_B *= scale;
	m_G *= scale;
	m_R *= scale;
	m_gradMagnVal *= scale;
	return *this;
}
PBASFeature& PBASFeature::operator /= (double scaleFactor)
{
	*this *= 1.0 / scaleFactor;
	return *this;
}
const PBASFeature PBASFeature::operator / (double scaleFactor) const
{
	PBASFeature result = *this;
	result /= scaleFactor;
	return result;
}
const float& PBASFeature::operator [] (int position) const
{
	CV_Assert(position >= 0 && position < NUM_FEATURES);
	switch (position)
	{
	case 0:
		return m_R;
	case 1:
		return m_G;
	case 2:
		return m_B;
	case 3:
		return m_gradMagnVal;
	}
}

PBASFeature::~PBASFeature()
{

}


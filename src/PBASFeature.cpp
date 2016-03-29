#include "PBASFeature.h"

float PBASFeature::fg_colorWeight = 0.8;


PBASFeature::PBASFeature()
{
	cv::RNG randomGenerator;
	m_B = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_R = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_G = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_gradX = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_gradY = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
}
PBASFeature::PBASFeature(int R, int G, int B, int avgB, int avgG, int avgR, int gMX, int gMY, float colorWeight)
{
	m_R = R; m_G = G; m_B = B;
	m_avgB = avgB;
	m_avgG = avgG;
	m_avgR = avgR;
	m_gradX = gMX;
	m_gradY = gMY;
}

PBASFeature::PBASFeature(const cv::Mat& color, int avgB, int avgG, int avgR, int gMX, int gMY, float colorWeight)
{
	CV_Assert(color.rows == 1 && color.cols == 1 && color.type() == CV_8UC3);
	m_B = color.ptr<uchar>(0)[0];
	m_G = color.ptr<uchar>(0)[1];
	m_R = color.ptr<uchar>(0)[2];
	m_avgB = avgB;
	m_avgG = avgG;
	m_avgR = avgR;
	m_gradX = gMX;
	m_gradY = gMY;
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
	m_avgB = other.m_avgB;
	m_avgG = other.m_avgG;
	m_avgR = other.m_avgR;
	m_gradX = other.m_gradX;
	m_gradY = other.m_gradY;
}

// calc functions
double PBASFeature::calcDistance(const PBASFeature& first, const PBASFeature& second)
{
	double result;

	result = std::abs(first.m_B - second.m_B) + std::abs(first.m_G - second.m_G) + std::abs(first.m_R - second.m_R);
		//+ (std::abs(first.m_avgB - second.m_avgB) + std::abs(first.m_avgG - second.m_avgG) + std::abs(first.m_avgR - second.m_avgR)));

	return result;

}
std::vector<PBASFeature> PBASFeature::calcFeatureMap(const cv::Mat& inputFrame)
{
	std::vector<PBASFeature> result;
	// get gradient magnitude map
	cv::Mat sobelX, sobelY, inputGray, gradMagnMap;

	cv::GaussianBlur(inputFrame, inputGray, cv::Size(5, 5), 2, 2, cv::BORDER_DEFAULT);
	cv::cvtColor(inputGray, inputGray, CV_BGR2GRAY);
	
	// Gradient Y
	Sobel(inputGray, sobelY, CV_64F, 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(sobelY, sobelY);

	// Gradient X
	Sobel(inputGray, sobelX, CV_64F, 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(sobelX, sobelX);

	sobelY.convertTo(sobelY, CV_8U);
	sobelX.convertTo(sobelX, CV_8U);

	int kernelSize = 7;
	cv::Mat avgKernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (kernelSize * kernelSize);
	cv::Mat avgMatB, avgMatG, avgMatR;
	cv::Mat channel[3];

	cv::split(inputFrame, channel);

	cv::filter2D(channel[0], avgMatB, CV_8U, avgKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(channel[1], avgMatG, CV_8U, avgKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(channel[2], avgMatR, CV_8U, avgKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	cv::Mat tmp(1, 1, CV_8UC3);

	for (int y = 0; y < inputFrame.rows; ++y)
	{
		for (int x = 0; x < inputFrame.cols; ++x)
		{

			tmp.ptr<uchar>(0)[0] = inputFrame.at<cv::Vec3b>(y, x)[0];
			tmp.ptr<uchar>(0)[1] = inputFrame.at<cv::Vec3b>(y, x)[1];
			tmp.ptr<uchar>(0)[2] = inputFrame.at<cv::Vec3b>(y, x)[2];



			result.push_back(PBASFeature(tmp, avgMatB.at<uchar>(y, x), avgMatG.at<uchar>(y, x), avgMatR.at<uchar>(y, x), sobelX.at<uchar>(y,x), sobelY.at<uchar>(y, x)));
		}


	}
	return result;

}
std::vector<PBASFeature> PBASFeature::calcFeatureMap(const cv::Mat& inputFrame, const cv::Mat& gradMagnMapX, const cv::Mat& gradMagnMapY)
{
	int kernelSize = 5;
	std::vector<PBASFeature> result;
	cv::Mat tmp(1, 1, CV_8UC3);
	
	cv::Mat avgKernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (kernelSize * kernelSize);
	cv::Mat avgMatB, avgMatG, avgMatR, smoothedInput;
	cv::Mat channel[3];

	
	cv::GaussianBlur(inputFrame, smoothedInput, cv::Size(7, 7), 2, 2, cv::BORDER_DEFAULT);
	cv::split(inputFrame, channel);


	cv::filter2D(channel[0], avgMatB, CV_8U, avgKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(channel[1], avgMatG, CV_8U, avgKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(channel[2], avgMatR, CV_8U, avgKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);


	for (int y = 0; y < inputFrame.rows; ++y)
	{
		for (int x = 0; x < inputFrame.cols; ++x)
		{

			tmp.ptr<uchar>(0)[0] = inputFrame.at<cv::Vec3b>(y, x)[0];
			tmp.ptr<uchar>(0)[1] = inputFrame.at<cv::Vec3b>(y, x)[1];
			tmp.ptr<uchar>(0)[2] = inputFrame.at<cv::Vec3b>(y, x)[2];

			result.push_back(PBASFeature(tmp, avgMatB.at<uchar>(y, x), avgMatG.at<uchar>(y, x), avgMatR.at<uchar>(y, x), gradMagnMapX.at<uchar>(y, x), gradMagnMapY.at<uchar>(y, x)));
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

			result += 0.5 * featureMap.at(y * width + x).m_gradX;
			result += 0.5 * featureMap.at(y * width + x).m_gradY;
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
	result.m_gradX = -m_gradX;
	result.m_gradY = -m_gradY;
	return result;
}
PBASFeature& PBASFeature::operator+= (const PBASFeature& other)
{
	m_B += other.m_B;
	m_G += other.m_G;
	m_R += other.m_R;
	m_gradX += other.m_gradX;
	m_gradY += other.m_gradY;
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
	m_gradX *= scale;
	m_gradY *= scale;
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
		return 	m_gradX;
	case 4:
		return 	m_gradY;
	}
}

PBASFeature::~PBASFeature()
{

}

double PBASFeature::calcGradSimilarity(const PBASFeature& first, const PBASFeature& second)
{

	double result = 0.5 * std::abs(first.m_gradX - second.m_gradX) + 0.5 * std::abs(first.m_gradY - second.m_gradY);

	return result;


}


#include "ColourFeature.h"

ColourFeature::ColourFeature()
{
	cv::RNG randomGenerator;
	m_B = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_R = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
	m_G = cv::saturate_cast<uchar>(randomGenerator.uniform(0, 255));
}

ColourFeature::ColourFeature(int R, int G, int B) : m_R(R), m_B(B), m_G(G)
{

}

ColourFeature::ColourFeature(const ColourFeature& other)
{
	copyOther(other);
}

ColourFeature::ColourFeature(const cv::Mat& other) // assume cv mat multichannel bgr at x,y
{
	CV_Assert(other.rows == 1 && other.cols == 1 && other.type() == CV_8UC3);
	m_B = other.ptr<uchar>(0)[0];
	m_G = other.ptr<uchar>(0)[1];
	m_R = other.ptr<uchar>(0)[2];
}

ColourFeature::~ColourFeature()
{
}

void ColourFeature::copyOther(const ColourFeature& other)
{
	m_B = other.m_B;
	m_R = other.m_R;
	m_G = other.m_G;
}


// operator overloading
ColourFeature& ColourFeature::operator= (const ColourFeature& other)
{
	if (this != &other)
	{
		copyOther(other);
	}
	return *this;
}

const ColourFeature ColourFeature::operator- () const
{
	ColourFeature result;
	result.m_B = -m_B;
	result.m_G = -m_G;
	result.m_R = -m_R;
	return result;
}
ColourFeature& ColourFeature::operator-= (const ColourFeature& other)
{
	*this += -other;
	return *this;
}
ColourFeature& ColourFeature::operator+= (const ColourFeature& other)
{
	m_B += other.m_B;
	m_G += other.m_G;
	m_R += other.m_R;
	return *this;
}
const ColourFeature ColourFeature::operator+ (const ColourFeature& other) const
{
	ColourFeature result = *this;
	result += other;
	return result;
}
const ColourFeature ColourFeature::operator- (const ColourFeature& other) const
{
	ColourFeature result = *this;
	result -= other;
	return result;
}
const ColourFeature ColourFeature::operator* (double scale) const
{
	ColourFeature result = *this;
	result *= scale;
	return result;
}
ColourFeature& ColourFeature::operator*= (double scale)
{
	m_B *= scale;
	m_G *= scale;
	m_R *= scale;
	return *this;
}
ColourFeature& ColourFeature::operator /= (double scaleFactor)
{
	*this *= 1.0 / scaleFactor;
	return *this;
}
const ColourFeature ColourFeature::operator / (double scaleFactor) const
{
	ColourFeature result = *this;
	result /= scaleFactor;
	return result;
}
const float& ColourFeature::operator [] (int position) const
{
	CV_Assert(position >= 0 && position <= 2);
	switch (position)
	{
	case 0:
		return m_R;
	case 1:
		return m_G;
	case 2:
		return m_B;
	}
}


double ColourFeature::calcDistance(const ColourFeature& first, const ColourFeature& second)
{
	// L2 distance
	double result;
	result = std::sqrt(std::pow(std::abs(first.m_B - second.m_B), 2) + std::pow(std::abs(first.m_G - second.m_G), 2) + std::pow(std::abs(first.m_R - second.m_R), 2));
	return result;
}


std::vector<ColourFeature> ColourFeature::calcFeatureMap(const cv::Mat& inputFrame)
{
	std::vector<ColourFeature> result;
	cv::Mat tmp(1, 1, CV_8UC3);

	for (int y = 0; y < inputFrame.rows; ++y)
	{
		for (int x = 0; x < inputFrame.cols; ++x)
		{

			tmp.ptr<uchar>(0)[0] = inputFrame.at<cv::Vec3b>(y, x)[0];
			tmp.ptr<uchar>(0)[1] = inputFrame.at<cv::Vec3b>(y, x)[1];
			tmp.ptr<uchar>(0)[2] = inputFrame.at<cv::Vec3b>(y, x)[2];

			result.push_back(ColourFeature(tmp));
		}


	}
	return result;
}
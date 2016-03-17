#pragma once
#include<opencv2\opencv.hpp>


class ColourFeature
{

protected:
	void copyOther(const ColourFeature& other);
	int m_R, m_G, m_B;
	static float fg_colorWeight;

public:
	ColourFeature(); // random initialization
	ColourFeature(int R, int G, int B); // copy constructor
	ColourFeature(const ColourFeature& other); // copy constructor
	ColourFeature(const cv::Mat& other); // assume cv mat multichannel bgr at x,y

	static double calcDistance(const ColourFeature& first, const ColourFeature& second, int Lnorm=2);
	static std::vector<ColourFeature> calcFeatureMap(const cv::Mat& inputFrame);
	static float getColorWeight() { return fg_colorWeight; }
	static void setColorWeight(float newVal) { CV_Assert(newVal >= 0 && newVal <= 1); fg_colorWeight = newVal; }
	~ColourFeature();

	static const int NUM_FEATURES = 3;

	// operator overloading
	ColourFeature& operator= (const ColourFeature& other);
	const ColourFeature operator- () const;
	ColourFeature& operator+= (const ColourFeature& other);
	const ColourFeature operator+ (const ColourFeature& other) const;
	ColourFeature& operator-= (const ColourFeature& other);
	const ColourFeature operator- (const ColourFeature& other) const;
	const ColourFeature operator* (double scale) const;
	ColourFeature& operator*= (double scale);
	ColourFeature& operator /= (double scaleFactor);
	const ColourFeature operator / (double scaleFactor) const;
	const float& operator [] (int position) const;
};




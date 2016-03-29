#pragma once
#include<opencv2\opencv.hpp>

class PBASFeature
{
protected:
	int m_gradX, m_gradY;
	int m_avgB, m_avgG, m_avgR;
	static float fg_colorWeight;
	void copyOther(const PBASFeature& other);

public:
	int m_R, m_G, m_B;

	static const int NUM_FEATURES = 4;

	PBASFeature(); // random initialization
	PBASFeature(int R, int G, int B, int avgB, int avgG, int avgR, int gMX, int gMY, float colorWeight = 0.8); // copy constructor
	PBASFeature(const PBASFeature& other); // copy constructor
	PBASFeature(const cv::Mat& color, int avgB, int avgG, int avgR, int gMX, int gMY, float colorWeight = 0.8); // assume cv mat multichannel bgr at x,y

	static double calcDistance(const PBASFeature& first, const PBASFeature& second);
	static double calcGradSimilarity(const PBASFeature& first, const PBASFeature& second);

	static std::vector<PBASFeature> calcFeatureMap(const cv::Mat& inputFrame);
	static std::vector<PBASFeature> calcFeatureMap(const cv::Mat& inputFrame, const cv::Mat& gradMagnMapX, const cv::Mat& gradMagnMapY);
	static float calcMeanGradMagn(const std::vector<PBASFeature>& featureMap, int height, int width);
	static float getColorWeight() { return fg_colorWeight; }
//	static void setColorWeight(float newVal) { CV_Assert(newVal >= 0 && newVal <= 1); fg_colorWeight = newVal; }


	~PBASFeature();

	// operator overloading
	PBASFeature& operator= (const PBASFeature& other);
	const PBASFeature operator- () const;
	PBASFeature& operator+= (const PBASFeature& other);
	const PBASFeature operator+ (const PBASFeature& other) const;
	PBASFeature& operator-= (const PBASFeature& other);
	const PBASFeature operator- (const PBASFeature& other) const;
	const PBASFeature operator* (double scale) const;
	PBASFeature& operator*= (double scale);
	PBASFeature& operator /= (double scaleFactor);
	const PBASFeature operator / (double scaleFactor) const;
	const float& operator [] (int position) const;
	
};


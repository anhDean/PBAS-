#pragma once
#include <opencv2\opencv.hpp>

class BackgroundFeature
{
	private:

	protected:
		cv::Mat m_color;
		float m_gradMagn;
		int m_regionSignatureHW;

		void copyOther(const BackgroundFeature& other);
		void clear();
		
	public:
		BackgroundFeature(); // init with random values
		~BackgroundFeature();
		BackgroundFeature(cv::Mat& color, float gradMagn, int regionSignature=0);
		BackgroundFeature(const BackgroundFeature& other); // copy constructor
		
		static double calcDistance(const BackgroundFeature& first, const BackgroundFeature& second);
		static std::vector<BackgroundFeature> calcFeatureMap(const cv::Mat& inputFrame, const cv::Mat& gradMagnMap, const cv::Mat& signatureHWMap);
		static const int NUM_FEATURES = 5;

		// operator overloading
		const BackgroundFeature operator- () const;
		BackgroundFeature& operator+= (const BackgroundFeature& other);
		const BackgroundFeature operator+ (const BackgroundFeature& other) const;
		BackgroundFeature& operator-= (const BackgroundFeature& other);
		const BackgroundFeature operator- (const BackgroundFeature& other) const;
		const BackgroundFeature operator* (double scale) const;
		BackgroundFeature& operator*= (double scale);
		BackgroundFeature& operator /= (double scaleFactor);
		const BackgroundFeature operator / (double scaleFactor) const;
		BackgroundFeature& operator= (const BackgroundFeature& other);
		const float& operator [] (int position) const;

};




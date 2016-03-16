#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

class FrameProcessor
{
	protected:
		int m_iteration;
		static const int BACKGROUNDVAL = 0, FOREGROUNDVAL = 255;
		cv::Mat m_gradMagnMap, m_lastResult, m_lastResultPP, m_currentResult, m_currentResultPP, m_noiseMap;
		virtual void updateNoiseMap()
		{
			cv::Mat tmp;
			cv::bitwise_xor(m_lastResult, m_currentResult, tmp);

			for (int y = 0; y < m_noiseMap.rows; ++y)
			{
				for (int x = 0; x < m_noiseMap.cols; ++x)
				{
					if (tmp.at<uchar>(y, x) == BACKGROUNDVAL ||
						(tmp.at<uchar>(y, x) == FOREGROUNDVAL && (m_lastResultPP.at<uchar>(y, x) == FOREGROUNDVAL || m_currentResultPP.at<uchar>(y, x) == FOREGROUNDVAL)))
					{
						m_noiseMap.at<double>(y, x) -= 0.2;
						m_noiseMap.at<double>(y, x) = (m_noiseMap.at<double>(y, x) < 0.0) ? 0 : m_noiseMap.at<double>(y, x);
					}
					else
						m_noiseMap.at<double>(y, x) += 1;
				}
			}
		}
		virtual void updateGradMagnMap(const cv::Mat& inputFrame)
		{
			cv::Mat sobelX, sobelY, inputGray;
			cv::cvtColor(inputFrame, inputGray, CV_BGR2GRAY);
			// features: gradient magnitude and direction and pixel intensities	
			cv::Sobel(inputGray, sobelX, CV_32F, 1, 0, 3, 1, 0.0); // get gradient magnitude for dx
			cv::Sobel(inputGray, sobelY, CV_32F, 0, 1, 3, 1, 0.0); // get gradient magnitude for dy
			cv::cartToPolar(sobelX, sobelY, m_gradMagnMap, sobelY, true); // convert cartesian to polar coordinates
		}
		
	public:
		FrameProcessor(){};
		virtual ~FrameProcessor() {};
		// processing method
		virtual void process(cv:: Mat &input, cv:: Mat &output)= 0;	
		virtual void process(cv:: Mat &input)= 0;
		virtual void resetProcessor() = 0;
		virtual std::auto_ptr<cv::Mat> getBackgroundDynamics() const = 0;
		virtual const cv::Mat& getGradMagnMap() const { return m_gradMagnMap; }
		virtual const cv::Mat& getNoiseMap() const
		{
			return m_noiseMap;
		}


		
};



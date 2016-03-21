#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

class FrameProcessor
{
	protected:
		int m_iteration;
		static const int BACKGROUNDVAL = 0, FOREGROUNDVAL = 255;
		cv::Mat m_gradMagnMap, m_lastResult, m_lastResultPP, m_currentResult, m_currentResultPP, m_noiseMap;
		virtual void updateGradMagnMap(const cv::Mat& inputFrame)
		{
			cv::Mat sobelX, sobelY, inputGray;

			cv::GaussianBlur(inputFrame, inputGray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
			cv::cvtColor(inputGray, inputGray, CV_BGR2GRAY);

			Sobel(inputGray, sobelX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
			convertScaleAbs(sobelX, sobelX);

			Sobel(inputGray, sobelY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
			convertScaleAbs(sobelY, sobelY);

			// Total Gradient (approximate)
			cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, m_gradMagnMap);
		}
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
						m_noiseMap.at<float>(y, x) -= m_noiseMap.at<float>(y, x) - 0.2;
					}
					else
						m_noiseMap.at<float>(y, x) += + 1;
					m_noiseMap.at<float>(y, x) = (m_noiseMap.at<float>(y, x) < 0) ? 0 : m_noiseMap.at<float>(y, x);
				}
			}
			normalizeMat(m_noiseMap);
		}


		
	public:
		FrameProcessor(){};
		virtual ~FrameProcessor() {};
		// processing method
		virtual void process(cv:: Mat &input, cv:: Mat &output)= 0;	
		virtual void process(cv:: Mat &input)= 0;
		virtual void resetProcessor() = 0;
		virtual const cv::Mat getBackgroundDynamics() const = 0;
		virtual const cv::Mat& getGradMagnMap() const { return m_gradMagnMap; }
		virtual const cv::Mat& getRawOutput() const = 0;
		virtual const cv::Mat& getNoiseMap() const
		{
			return m_noiseMap;
		}
		static void normalizeMat(cv::Mat input)
		{
			// normalize between 0 and 1
			double min, max;
			// normalize gradient magnmap
			cv::minMaxIdx(input, &min, &max);
			input.convertTo(input, CV_32F, 1.0 / (float)max);
		}
		
};



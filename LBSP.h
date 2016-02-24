#pragma once
#include <opencv2/core/core.hpp>
#include <vector>
struct binaryStr
{
	int length = 16;
	std::vector<bool> x;
};

class LBSP
{
	private:
		cv::Mat m_im;
		std::vector<binaryStr> m_binaryStrArray;
		float Thresh;

	public:
		LBSP();
		LBSP(const cv::Mat& input); // copy constructor

		binaryStr getBinaryStrXY(const int& j, const int& i) const;		// returns binary string struct at position X (col),Y (row)
		binaryStr operator-(const LBSP& b); // returns inter-frame binary string (difference between two binary strings)
		binaryStr calcDiffBinaryStr(const binaryStr& a, const binaryStr& b) const; // calculates difference string between two  binary string representations
		~LBSP();
};


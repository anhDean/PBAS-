#include "LBSP.h"

const std::vector<int> LBSP::LBSP_PATTERN_ORDER_X = { 0, 2, 4, 1, 2, 3, 0, 1, 3, 4, 1, 2, 3, 0, 2, 4 }; // X-Coordinate order for LBSP processing 5X5 image patch
const std::vector<int> LBSP::LBSP_PATTERN_ORDER_Y = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4 }; // Y-Coordinate order for LBSP processing
const int LBSP::LBSP_NUM = 16;

LBSP::LBSP(const cv::Mat& input, float intraThresh, float interThresh) : m_nRows(input.rows), m_nCols(input.cols), m_intraThresh(intraThresh), m_LBSPArray(input.rows * input.cols), m_interTresh(0.3)
{
	CV_Assert(input.type() == CV_8UC1 || input.type() == CV_8UC3);
	setLBSPArray(input);
}
void LBSP::init(const cv::Mat& input)
{
	
	CV_Assert(input.type() == CV_8UC1 || input.type() == CV_8UC3);
	if (!m_LBSPArray.empty()) m_LBSPArray.clear();
	m_intraThresh = 25;
	m_interTresh = 0.3;
	m_nCols = input.cols;
	m_nRows = input.rows;
	m_LBSPArray.resize(m_nCols * m_nRows);
	setLBSPArray(input);

}

LBSP::LBSP(const cv::Mat& input, const cv::Mat& formerInput, float intraThresh, float interThresh) : m_nRows(input.rows), m_nCols(input.cols), m_intraThresh(intraThresh), m_LBSPArray(input.rows * input.cols), m_interTresh(interThresh)
{
	CV_Assert(input.type() == CV_8UC1 || input.type() == CV_8UC3);
	setLBSPArray(input, formerInput);
}




LBSP& LBSP::operator=(const LBSP& b) // returns matrix (in vector representation) of inter-frame LBSP
{
	if (this != &b)
	{
		m_LBSPArray.clear();
		m_nRows = b.m_nRows;
		m_nCols = b.m_nCols;
		m_intraThresh = b.m_intraThresh;
		m_interTresh = b.m_interTresh;
		setLBSPArray(b.m_LBSPArray);
	}
	return *this;
}
LBSP::~LBSP()
{
	m_LBSPArray.clear();
}
// setters
void LBSP::setLBSPArray(const std::vector<BinaryStr>& lbsp_array)
{
	CV_Assert(lbsp_array.size() == m_nRows * m_nCols && lbsp_array[0].size() == LBSP_NUM);
	m_LBSPArray.clear();
	for (int it = 0; it < lbsp_array.size(); ++it)
	{
		m_LBSPArray.push_back(lbsp_array[it]);
	}
}
void LBSP::setLBSPArray(const cv::Mat &input)
{	
	CV_Assert((input.type() == CV_8UC1) || (input.type() == (CV_8UC3)) && input.rows == m_nRows && input.cols == m_nCols && m_LBSPArray.size() == m_nCols * m_nRows);

	// pad image to get safe patches
	cv::Mat padded_input = padMat(input);

	for (int y = 0; y < m_nRows; ++y)
	{
		for (int x = 0; x < m_nCols; ++x)
		{
			m_LBSPArray.at(y* m_nCols + x) = calcLBSPXY(padded_input, x, y);
		}
	}

}
void LBSP::setLBSPArray(const cv::Mat &input, const cv::Mat &formerInput)
{
	CV_Assert((input.type() == CV_8UC1) || (input.type() == (CV_8UC3)) && input.rows == m_nRows && input.cols == m_nCols && m_LBSPArray.size() == m_nCols * m_nRows);

	// pad image to get safe patches
	cv::Mat padded_input = padMat(input);

	for (int y = 0; y < m_nRows; ++y)
	{
		for (int x = 0; x < m_nCols; ++x)
		{
			m_LBSPArray.at(y* m_nCols + x) = calcLBSPXY(padded_input, formerInput, x, y);
		}
	}


}
BinaryStr LBSP::calcLBSPXY(const cv::Mat &im, const int &x, const int &y) const
{
	const int padding = 2, centerX = 2, centerY = 2; // assume 5x5 patches
	CV_Assert((im.type() == CV_8UC1 || im.type() == CV_8UC3) && (m_nCols + 2 * padding == im.cols)  && (m_nRows + 2 * padding == im.rows)); // assume padding

	bool tmp;
	BinaryStr out;
	cv::Mat patchXY;

	patchXY = im(cv::Range(y, y + padding * 2 + 1), cv::Range(x, x + padding * 2 + 1)).clone(); // range [0,N) : N is not included!

	if (im.type() == CV_8UC1)
	{
		uchar l1_diff;
		uchar centerVec = patchXY.at<uchar>(centerY, centerX);

		for (int i = 0; i < LBSP_PATTERN_ORDER_X.size(); ++i)
		{
			l1_diff = std::abs(patchXY.at<uchar>(LBSP_PATTERN_ORDER_Y[i], LBSP_PATTERN_ORDER_X[i]) - centerVec);

			tmp = (l1_diff > m_intraThresh) ? true : false;  // calculate value for processed pixel
			out.push_back(tmp);
		}
	}
	else if (im.type() == CV_8UC3)
	{
		cv::Mat l1_diff;
		cv::Vec3b centerVec = patchXY.at<cv::Vec3b>(centerY, centerX);

		for (int i = 0; i < LBSP_PATTERN_ORDER_X.size(); ++i)
		{
			cv::absdiff(centerVec, patchXY.at<cv::Vec3b>(LBSP_PATTERN_ORDER_Y[i], LBSP_PATTERN_ORDER_X[i]), l1_diff);
			tmp = (cv::mean(l1_diff)[0] > m_intraThresh) ? true : false;
			
			out.push_back(tmp);
		}
	}

	return out;
}
BinaryStr LBSP::calcLBSPXY(const cv::Mat &im, const cv::Mat &old_im, const int &x, const int &y) const
{
	const int padding = 2, centerX = 2, centerY = 2; // assume 5x5 patches
	CV_Assert((im.type() == CV_8UC1 || im.type() == CV_8UC3) && (m_nCols + 2 * padding == im.cols) && (m_nRows + 2 * padding == im.rows)); // assume padding

	bool tmp;
	BinaryStr out;
	cv::Mat patchXY;

	patchXY = im(cv::Range(y, y + padding * 2 + 1), cv::Range(x, x + padding * 2 + 1)).clone(); // range [0,N) : N is not included!

	if (im.type() == CV_8UC1)
	{
		uchar l1_diff;
		uchar centerVec = old_im.at<uchar>(y, x);

		for (int i = 0; i < LBSP_PATTERN_ORDER_X.size(); ++i)
		{
			l1_diff = std::abs(patchXY.at<uchar>(LBSP_PATTERN_ORDER_Y[i], LBSP_PATTERN_ORDER_X[i]) - centerVec);

			tmp = (l1_diff > m_interTresh * centerVec) ? true : false;  // calculate value for processed pixel
			out.push_back(tmp);
		}
	}
	else if (im.type() == CV_8UC3)
	{
		cv::Mat l1_diff;
		cv::Vec3b centerVec = old_im.at<cv::Vec3b>(y, x);

		for (int i = 0; i < LBSP_PATTERN_ORDER_X.size(); ++i)
		{
			cv::absdiff(centerVec, patchXY.at<cv::Vec3b>(LBSP_PATTERN_ORDER_Y[i], LBSP_PATTERN_ORDER_X[i]), l1_diff);
			tmp = (cv::mean(l1_diff)[0] > m_interTresh * cv::mean(centerVec)[0]) ? true : false;

			out.push_back(tmp);
		}
	}

	return out;
}



void LBSP::setIntraThreshold(float newVal)
{
	m_intraThresh = newVal;
}
// getters
const BinaryStr& LBSP::getLBSPXY(const int& x, const int&y) const // returns binary string struct at position X (col),Y (row)
{
	CV_Assert(!m_LBSPArray.empty());
	return m_LBSPArray[y * m_nCols  + x ];
}
const float& LBSP::getIntraThreshold() const
{
	return m_intraThresh;
}
const int& LBSP::getRows() const
{
	return m_nRows;
}
const int& LBSP::getCols() const
{
	return m_nCols;
}
// display functions
void LBSP::displayLBSPXY(const int &x, const int &y) const {

	for (int i = 0; i< m_LBSPArray.at(y * m_nCols + x).size(); ++i)
	{
		std::cout << m_LBSPArray.at(y * m_nCols + x)[i];
	}
	std::cout << std::endl;
}
// static member functions
cv::Mat LBSP::calcLBSPArrayDiff(const LBSP&a, const LBSP &b) // returns hammingweight array of bitwise xored lbsp arrays 
{
	int aCols = a.getCols(), bCols = b.getCols();
	int aRows = a.getRows(), bRows = b.getRows();
	cv::Mat result(aRows, aCols, CV_8UC1, cv::Scalar(0));
	std::vector<BinaryStr> tmp;
	CV_Assert(aCols == bCols && aRows == bRows);
	for (int y = 0; y < aRows; ++y)
	{
		for (int x = 0; x < aCols; ++x)
		{
			tmp.push_back(calcLBSPDiff(a.getLBSPXY(x, y), b.getLBSPXY(x, y)));
		}
	}
	LBSP2HWArray(tmp, aRows, aCols).copyTo(result);
	return result;
}
cv::Mat LBSP::LBSP2HWArray(const std::vector<BinaryStr>&a, int rows, int cols)
{
	CV_Assert(a.size() == rows * cols);
	int count;
	cv::Mat out(rows, cols, CV_8UC1, cv::Scalar(0));

	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			count = 0;
			for (int k = 0; k < LBSP_NUM; ++k)
			{
				if (a.at(y * cols + x).at(k) == true)
					count += 1;
			}
			out.at<uchar>(y,x) = count;
		}
	}
	return out;
}

BinaryStr LBSP::calcLBSPDiff(const BinaryStr& a, const BinaryStr& b)  // calculates difference string between two  binary string representations (pixel-wise inter-frame LBSP)
{
	BinaryStr result(16);
	CV_Assert(a.size() == b.size());
	for (int i = 0; i< a.size(); ++i) {
		result.at(i) =a.at(i) ^ b.at(i);
	}
	return result;
}
cv::Mat LBSP::padMat(const cv::Mat& input, int padding)
{
	cv::Mat padded_input;
	padded_input.create(input.rows + 2 * padding, input.cols + 2 * padding, input.type());
	padded_input.setTo(cv::Scalar::all(0));
	input.copyTo(padded_input(cv::Rect(padding, padding, input.cols, input.rows)));
	return padded_input;
}
void LBSP::displayPatchXY(cv::Mat in, int upper_leftX, int upper_leftY, int patchSize, bool disp)
{
	cv::Mat padded_input = padMat(in);
	std::cout << padded_input(cv::Rect(upper_leftX, upper_leftY, patchSize, patchSize)) << std::endl;
	if (disp)
	{
		cv::namedWindow("patch", CV_WINDOW_AUTOSIZE);
		cv::imshow("patch", in(cv::Rect(upper_leftX, upper_leftY, patchSize, patchSize)));
	}
}
void LBSP::displayLBSP(const BinaryStr &x) {
	for (int i = 0; i< x.size(); ++i) {
		std::cout << x.at(i);
	}
	std::cout << std::endl;
}
void LBSP::displayLBSPPatch(const BinaryStr &x) {
	cv::Mat out(5, 5, CV_8UC1, cv::Scalar(0));
	for (int i = 0; i< x.size(); ++i) {
		out.at<uchar>(LBSP_PATTERN_ORDER_Y[i], LBSP_PATTERN_ORDER_X[i]) = cv::saturate_cast<uchar>(x[i]);
	}
	std::cout << out << std::endl;
}


const std::vector<BinaryStr>& LBSP::getLBSPArray() const
{
	return m_LBSPArray;
}

const float& LBSP::getInterThreshold() const
{
	return m_interTresh;
}


void LBSP::setInterThreshold(float newVal)
{
	m_interTresh = newVal;
}
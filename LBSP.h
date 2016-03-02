#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>


// include guard for logger
#ifndef LOGGER_IDENT
#define LOGGER_IDENT
#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__<< "(" << __LINE__ << ")" << (x) << std::endl
#else
#define LOG_MESSAGE(x)
#endif
#endif

typedef std::vector<bool> BinaryStr; // TODO: make binarystr to struct with reference value



class LBSP
{
private:
	// data member
	int m_nRows, m_nCols;
	float m_thresh;
	std::vector<BinaryStr> m_LBSPArray;	// represents matrix with LBSPs
										// static data member
	static const int LBSP_NUM; // length of binary pattern
	static const std::vector<int> LBSP_PATTERN_ORDER_X;
	static const std::vector<int> LBSP_PATTERN_ORDER_Y;
	// member funtions
	BinaryStr calcLBSPXY(const cv::Mat &im, const int &x, const int &y) const; // calculates LBSP at given x,y coordinate
	

public:
	//constructor, destructor
	LBSP();
	LBSP(const LBSP& input); // copy constructor
	LBSP& operator=(const LBSP& b); // assignment operator
	LBSP(const cv::Mat& input, const float& thresh = 0.3); // constructor: generates matrix of LBSP pattern
	void init(const cv::Mat& input);
	~LBSP();
	// getters
	const int& getCols() const;
	const int& getRows() const;
	const float& getThreshold() const;
	const BinaryStr& getLBSPXY(const int& x, const int&y) const;
	// setters
	void setThreshold(const float& newVal);
	void setLBSPArray(const std::vector<BinaryStr>& lbsp_array);
	void setLBSPArray(const cv::Mat &input); // calculates LBSParray and writes into member 
	// display function
	void displayLBSPXY(const int &x, const int &y) const;


	//TODO: updatelbsp(frame_old, frame_new)
	//TODO: calcinterframeLBSPXY
	//TODO: make descriptor (reference value + signature)

	//static member functions
	static cv::Mat calcLBSPArrayDiff(const LBSP&a, const LBSP &b); // returns hammingweight array of difference of two LBSP instances
	static cv::Mat LBSP2HWArray(const std::vector<BinaryStr>&a, int rows, int cols); // returns array with hamming weights of LBSP patterns
	static BinaryStr calcLBSPDiff(const BinaryStr& a, const BinaryStr& b);
	static void displayLBSP(const BinaryStr &x);
	static void displayLBSPPatch(const BinaryStr &x);
	static cv::Mat padMat(const cv::Mat& input, int padding = 2);
	static void displayPatchXY(cv::Mat in, int upper_leftX, int upper_leftY, int patchSize = 5, bool disp = false);
};



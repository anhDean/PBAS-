#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

typedef std::vector<bool> BinaryStr; // TODO: make binarystr to struct with reference value

class LBSP
{
private:
	// data member
	int m_nRows, m_nCols;
	float m_intraThresh, m_interTresh;
	std::vector<BinaryStr> m_LBSPArray;	// represents matrix with LBSPs
	
	// static data member
	static const int LBSP_NUM; // length of binary pattern
	static const std::vector<int> LBSP_PATTERN_ORDER_X;
	static const std::vector<int> LBSP_PATTERN_ORDER_Y;
	
	// member funtions
	BinaryStr calcLBSPXY(const cv::Mat &im, const int &x, const int &y) const; // calculates LBSP at given x,y coordinate
	BinaryStr calcLBSPXY(const cv::Mat &im, const cv::Mat &old_im, const int &x, const int &y) const;
	void init(const cv::Mat& input);

public:
	//constructor, destructor
	LBSP(const cv::Mat& input, float intraThresh = 25, float interThresh = 0.3); // constructor: generates matrix of LBSP patternl
	LBSP(const cv::Mat& input, const cv::Mat& formerInput, float intraThresh = 25, float interThresh = 0.3); // constructor: generates inter frame lbsp with reference value from old frame
	LBSP& operator=(const LBSP& b); // assignment operator
	~LBSP();

	// getters
	const int& getCols() const;
	const int& getRows() const;
	const float& getIntraThreshold() const;
	const float& getInterThreshold() const;
	const std::vector<BinaryStr>& getLBSPArray() const;
	const BinaryStr& getLBSPXY(const int& x, const int&y) const;
	// setters
	void setIntraThreshold(float newVal);
	void setInterThreshold(float newVal);
	void setLBSPArray(const std::vector<BinaryStr>& lbsp_array);
	void setLBSPArray(const cv::Mat &input); // calculates LBSParray and writes into member
	void setLBSPArray(const cv::Mat &input, const cv::Mat &formerInput); // calculates LBSParray and writes into member 
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



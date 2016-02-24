#include "LBSP.h"


LBSP::LBSP(const cv::Mat& input) 
{
	m_im = input.clone();
}

LBSP::LBSP()
{
}


LBSP::~LBSP()
{
}

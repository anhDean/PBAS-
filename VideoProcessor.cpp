#include "VideoProcessor.h"

VideoProcessor::VideoProcessor(const std::string& filename) : m_callIt(true), m_delay(0),m_fnumber(0), m_stop(false), m_frameToStop(-1)
{
	cv::Mat temp;
	m_capture.release();
	m_capture.open(filename); //filename
	m_capture.read(temp);
	m_frameSize = temp.size();
}

VideoProcessor::VideoProcessor(const std::vector<std::string>& imgs) : m_callIt(true), m_delay(0), m_fnumber(0), m_stop(false), m_frameToStop(-1), m_images(imgs), m_itImg(m_images.begin())
{
	m_capture.release();
}

VideoProcessor::~VideoProcessor(void)
{
	m_capture.release();
}



// to display the processed frames
void VideoProcessor::displayInput(const std::string & windowName) const {
	m_windowNameInput= windowName;
	cv::namedWindow(m_windowNameInput);
}
// to display the processed frames
void VideoProcessor::displayOutput(const std::string & windowName) const {
	m_windowNameOutput= windowName;
	cv::namedWindow(m_windowNameOutput);
}
// do not display the processed frames
void VideoProcessor::stopDisplay() const {
	cv::destroyWindow(m_windowNameInput);
	cv::destroyWindow(m_windowNameOutput);
	m_windowNameInput.clear();
	m_windowNameOutput.clear();
}

// to grab (and process) the frames of the sequence
void VideoProcessor::run() {
	// current frame
	cv::Mat frame, output;
	// if no m_capture device has been set
	if (!isOpened()) return;
	m_stop = false;
	
	while (!isStopped()) {

		// read next frame if any
		if (!readNextFrame(frame))
			break;

		// display input frame
		/*
		if (m_windowNameInput.length()!=0)
			cv::imshow(m_windowNameInput,frame);
		*/

		// ** calling the process function or method **
		if (m_callIt) 
		{
			// process the frame
			if (process) // if call back function
				process(frame, output);

			else if (m_frameProcessor)
				// if class interface instance
				m_frameProcessor->process(frame,output);
			// increment frame number
			++m_fnumber;
		} 
		
		else {
			output = frame;
		}
		// ** write output sequence **
		if (m_outputFile.length()!=0)
			writeNextFrame(output);

		/*
		// display output frame
		if (m_windowNameOutput.length()!=0 && output.empty() !=0)
			cv::imshow(m_windowNameOutput,output);

		// introduce a delay
		if (m_delay>=0 && cv::waitKey(m_delay)>=0)
			setStop();

		// check if we should stop
		if (m_frameToStop>=0 &&
			getFrameNumber()== m_frameToStop)
			setStop();
		*/
	}
}
// Stop the processing
void VideoProcessor::setStop() {
	m_stop= true;
}
// Is the process stopped?
bool VideoProcessor::isStopped() {
	return m_stop;
}
// Is a m_capture device opened?
bool VideoProcessor::isOpened() {
	return m_capture.isOpened() || !m_images.empty();
}


void VideoProcessor::setDelay(int d) {
	// set a delay between each frame
	// 0 means wait at each frame
	// negative means no delay	
	m_delay= d;
}


// to get the next frame
// could be: video file or camera
bool VideoProcessor::readNextFrame(cv::Mat& frame) {
	if (m_images.size()==0)
	{
		return m_capture.read(frame); // writes into input argument
	}
	else 
	{
		if (m_itImg != m_images.end()) 
		{
			frame = cv::imread(*m_itImg); // writes into input argument
			m_itImg++;
			return frame.data != 0;
		} 
		else 
		{
			return false;
		}
	}
}

// process callback to be called
void VideoProcessor::setCallProcess() {
	m_callIt= true;
}

// do not call process callback
void VideoProcessor::unsetCallProcess() {
	m_callIt= false;
}

void VideoProcessor::stopAtFrameNo(long frame) {
	m_frameToStop= frame;
}
// return the frame number of the next frame
long VideoProcessor::getFrameNumber() const{
	// get info of from the m_capture device
	long fnumber= cv::saturate_cast<long>(
		m_capture.get(CV_CAP_PROP_POS_FRAMES));
	return fnumber;
}

double VideoProcessor::getFrameRate() const
{
	return m_capture.get(CV_CAP_PROP_FPS);
}

long VideoProcessor::getTotalNumberOfFrames() const
{
	return m_capture.get(CV_CAP_PROP_FRAME_COUNT);
}


// set the instance of the class that
// implements the FrameProcessor interface
void VideoProcessor::setFrameProcessor(FrameProcessor* frameProcessorPtr)
{
	// invalidate callback function
	process = 0;
	// this is the frame processor instance
	// that will be called
	m_frameProcessor = frameProcessorPtr;
	setCallProcess();
}

// set the callback function that
// will be called for each frame
void VideoProcessor::setFrameProcessor(void (*frameProcessingCallback)(cv::Mat&, cv::Mat&)) {

	// invalidate frame processor class instance
	m_frameProcessor= 0;
	// this is the frame processor function that
	// will be called
	process= frameProcessingCallback;
	setCallProcess();
}






/*###################################################################################################################*/
/*WRITE VIDEO*/
/*###################################################################################################################*/
// set the output video file
// by default the same parameters than
// input video will be used
bool VideoProcessor::setOutput(std::string filename, int codec, double framerate,  bool isColor) 
{
	char c[4];
	int codec_ = codec;
	double frRate = framerate;
	m_outputFile = filename;
	m_extension.clear();

	if (frRate ==0.0)
	{
		frRate = getFrameRate(); // same as input
	}
	
	// use same codec as input
	if (codec_ ==0)
	{
		codec_ = getCodec(c);
	}

	// Open output video
	return m_writer.open(m_outputFile, codec_, frRate, getFrameSize(), // frame size 
		isColor); // color video?
	
}

// to write the output frame
// could be: video file or images
void VideoProcessor::writeNextFrame(cv::Mat& frame) {
	if (m_extension.length())
	{ 
		// then we write images
		std::stringstream ss;
		
		// compose the output filename
		ss << m_outputFile << std::setfill('0') << std::setw(m_digits) << m_currentIndex++ << m_extension;
		cv::imwrite(ss.str(),frame);
	} 
	else 
	{ 
		m_writer.write(frame);
	}
	m_frameSize = frame.size();
}

// set the output as a series of image files
// extension must be ".jpg", ".bmp" ...
bool VideoProcessor::setOutput(std::string filename, const std::string &ext, // image file extension
			   int numberOfDigits, int startIndex) 
{ 
	if (numberOfDigits<0)
		return false;
	// start index
	// number of digits must be positive
	// filenames and teirh common extension
	m_outputFile = filename;
	m_extension= ext;
	// number of digits in the file numbering scheme
	m_digits= numberOfDigits;
	// start numbering at this index
	m_currentIndex= startIndex;
	return true;
}

// get the codec of input video
int VideoProcessor::getCodec(char codec[4]) 
{
	
	// undefined for vector of images
	if (m_images.size()!=0) return -1;
	union { // data structure for the 4-char code
		int value;
		char codec[4]; } returned;
		// get the code
		returned.value= static_cast<int>(
			m_capture.get(CV_CAP_PROP_FOURCC));
		// get the 4 characters
		codec[0]= returned.codec[0];
		codec[1]= returned.codec[1];
		codec[2]= returned.codec[2];
		codec[3]= returned.codec[3];
		// return the int value corresponding to the code
		return returned.value;
}

const cv::Size& VideoProcessor::getFrameSize() const
{
	return m_frameSize;
}
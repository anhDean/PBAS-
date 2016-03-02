#pragma once
#include <filesystem>
#include<vector>
#include<string>
#include<algorithm>
#include<opencv2/opencv.hpp>


namespace fs =  std::tr2::sys;

class FileHandler
{
private: 
	int m_root_depth; // levels until the dataset pictures are reached from root folder
	std::string m_inputSuffix, m_inputPrefix, m_outputSuffix, m_outputPrefix;
	std::string m_input_root, m_output_root;
	std::vector<std::string> m_inputFolders;
	std::vector<std::string> m_outputFolders;


	std::vector<std::string> getInputFoldernames() const;
	std::vector<std::string> getNextPathname(std::vector<std::string> in) const;
	void setOutputFolders();

	bool replaceSubstring(std::string& src, const std::string& old_str, const std::string& new_str);
	void createDirectories() const;
	std::string getOutputFilename(const std::string& inputFilename);


public:
	// TODO: new constructor for videos -> use cv:: videocapture FileHandler(captureobject, outputdirectory)
	// TODO: process_video member function -> for processing videos cv::Mat getFrame() -> process -> write
	template<class FrameProcessorClass>
	bool process_folder(std::string inputFolder, std::string outputFolder, FrameProcessorClass& processor) const;
	
	template<class FrameProcessorClass>
	bool process_folder(int idx, FrameProcessorClass& processor) const;

	FileHandler(const std::string& inpRt, const std::string& outRt, int rt_depth);
	const int& getFolderCount() const;
	const std::vector<std::string>& getInputFolders() const;
	const std::vector<std::string>& getOutputFolders() const;
	bool setInputSuffix(const std::string& newVal);  // must contain ".", e.g. ".jpg"
	bool setOutputSuffix(const std::string& newVal); // must contain ".", e.g. ".jpg"
	void setInputPrefix(const std::string& newVal);
	void setOutputPrefix(const std::string& newVal);

};

// template member functions defined in header

template<class FrameProcessorClass>
bool FileHandler::process_folder(std::string inputFolder, std::string outputFolder, FrameProcessorClass& processor) const
{
	// get file name
	std::string tmp_inputFile, tmp_outputFile;
	cv::Mat input, result;

	for (fs::directory_iterator it(inputFolder), end; it != end; ++it)
	{

		if (fs::is_regular_file(it->path()) && (it->path().filename().string().find(m_inputSuffix) != std::string::npos))
		{
			tmp_inputFile = it->path().filename().string();
			tmp_outputFile = getOutputFilename(tmp_inputFile);
		}
		input = cv::imread(tmp_inputFile, CV_LOAD_IMAGE_COLOR);
		result.create(input.size(), CV_8U);
		processor->process(input, result);

		cv::imwrite(outputFolder + "\\" + tmp_outputFile, result);
	}

	return true;
}

template<class FrameProcessorClass>
bool FileHandler::process_folder(int idx, FrameProcessorClass& processor) const
{
	// get file name
	std::string tmp_inputFile, tmp_outputFile;
	cv::Mat input, result;

	for (fs::directory_iterator it(m_inputFolders[idx]), end; it != end; ++it)
	{

		if (fs::is_regular_file(it->path()) && (it->path().filename().string().find(m_inputSuffix) != std::string::npos))
		{
			tmp_inputFile = it->path().filename().string();
			tmp_outputFile = getOutputFilename(tmp_inputFile);
		}
		input = cv::imread(tmp_inputFile, CV_LOAD_IMAGE_COLOR);
		result.create(input.size(), CV_8U);
		processor->process(input, result);

		cv::imwrite(m_outputFolders[idx] + "\\" + tmp_outputFile, result);
	}
	return true;
}

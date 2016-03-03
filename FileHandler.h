#pragma once
#include <filesystem>
#include<vector>
#include<string>
#include<algorithm>
#include<opencv2/opencv.hpp>
#include<Windows.h> // used Only for sleep function in createDirectories


namespace fs =  std::tr2::sys;

class FileHandler
{
private: 
	int m_root_depth; // levels until the dataset pictures are reached from root folder
	bool m_showProcessing;
	std::string m_inputSuffix, m_inputPrefix, m_outputSuffix, m_outputPrefix;
	std::string m_input_root, m_output_root;
	std::vector<std::string> m_inputFolders;
	std::vector<std::string> m_outputFolders;
	void createDirectories() const;

	std::vector<std::string> getInputFoldernames() const;
	std::vector<std::string> getNextPathname(std::vector<std::string> in) const;
	void setOutputFolders();
	bool replaceSubstring(std::string& src, const std::string& old_str, const std::string& new_str);
	std::string getOutputFilename(const std::string& inputFilename);



public:
	FileHandler(const std::string& inpRt, const std::string& outRt, int rt_depth, bool setupDirectories);
	// TODO: new constructor for videos -> use cv:: videocapture FileHandler(captureobject, outputdirectory)
	// TODO: process_video member function -> for processing videos cv::Mat getFrame() -> process -> write
	template<class FrameProcessorClass>
	bool process_folder(std::string inputFolder, std::string outputFolder, FrameProcessorClass* processor);
	
	template<class FrameProcessorClass>
	bool process_folder(int idx, FrameProcessorClass* processor);

	
	const size_t& getFolderCount() const;
	const std::vector<std::string>& getInputFolders() const;
	const std::vector<std::string>& getOutputFolders() const;
	const bool& getDisplayFlag()const;


	bool setInputSuffix(const std::string& newVal);  // must contain ".", e.g. ".jpg"
	bool setOutputSuffix(const std::string& newVal); // must contain ".", e.g. ".jpg"
	void setInputPrefix(const std::string& newVal);
	void setOutputPrefix(const std::string& newVal);
	void setDisplayFlag(bool flag);
};

// template member functions defined in header

template<class FrameProcessorClass>
bool FileHandler::process_folder(std::string inputFolder, std::string outputFolder, FrameProcessorClass* processor) 
{
	// get file name
	std::string tmp_inputFile, tmp_outputFile, windowName = outputFolder;
	cv::Mat input, output;

	for (fs::directory_iterator it(inputFolder), end; it != end; ++it)
	{

		if (fs::is_regular_file(it->path()) && (it->path().filename().string().find(m_inputSuffix) != std::string::npos))
		{
			tmp_inputFile = it->path().filename().string();
			tmp_outputFile = getOutputFilename(tmp_inputFile);
		}
		input = cv::imread(it->path().string(), CV_LOAD_IMAGE_COLOR);
		output.create(input.size(), CV_8U);
		processor->process(input, output);

		if(m_showProcessing)
		{
			cv::namedWindow(windowName);
			cv::imshow(windowName, output);
			if (cv::waitKey(1) == 27)  // 27 = ESC
				break;
		}
		cv::imwrite(outputFolder + "\\" + tmp_outputFile, output);
	}
	processor->resetProcessor();
	input.release();
	output.release();
	cv::destroyWindow(windowName);
	return true;
}

template<class FrameProcessorClass>
bool FileHandler::process_folder(int idx, FrameProcessorClass* processor)
{
	std::string inputFolder = m_inputFolders[idx];
	std::string outputFolder = m_outputFolders[idx];
	process_folder(inputFolder,  outputFolder, processor):
	return true;
}



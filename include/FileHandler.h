#pragma once
#include<filesystem>
#include<vector>
#include<string>
#include<algorithm>
#include<opencv2/opencv.hpp>
#include "FrameProcessor.h"
#include<Windows.h> // used Only for sleep function in createDirectories

namespace fs = std::tr2::sys;

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
	FileHandler();
	FileHandler(const std::string& inpRt, const std::string& outRt, int rt_depth, bool setupDirectories);
	// TODO: new constructor for videos -> use cv:: videocapture FileHandler(captureobject, outputdirectory)
	// TODO: process_video member function -> for processing videos cv::Mat getFrame() -> process -> write
	bool process_folder(std::string inputFolder, std::string outputFolder, FrameProcessor* processor);
	bool process_folder(int idx, FrameProcessor* processor);

	
	const size_t& getFolderCount() const;
	const std::vector<std::string>& getInputFolders() const;
	const std::vector<std::string>& getOutputFolders() const;
	const bool& getDisplayFlag()const;


	bool setInputSuffix(const std::string& newVal);  // must contain ".", e.g. ".jpg"
	bool setOutputSuffix(const std::string& newVal); // must contain ".", e.g. ".jpg"
	void setInputPrefix(const std::string& newVal);
	void setOutputPrefix(const std::string& newVal);
	void setDisplayFlag(bool flag);

	bool createDirectory(const std::string dir) const;
};

#pragma once

#include "hed.h"

/***
 * A line with start point (x1,y1) and end point (x2,y2)
 */
struct Line {
    int x1;
    int x2;
    int y1;
    int y2;
    float length;
};


class Detector {
public:
    /***
	 * @brief constructor
	 * @param model_path - path of the TorchScript weight file
	 * @param device_type - inference with CPU/GPU
	 */
	Detector(const char* model_path, const torch::DeviceType& device_type);

    /***
	 * @brief inference
	 * @param img - input image
	 * @return vector of lines in decreasing order of length
	 */
	std::vector<Line> getEdges(const cv::Mat& img);
private:
    HED model;

};
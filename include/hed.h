#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <memory>

class HED {
public:
    /***
	 * @brief constructor
	 * @param model_path - path of the TorchScript model file
	 * @param device_type - inference with CPU/GPU
	 */
	HED(const char* model_path, const torch::DeviceType& device_type);

    /***
	 * @brief inference
	 * @param img - input image
	 * @param output - tensor to hold inference result
	 */
	void Run(const cv::Mat& img, torch::Tensor& output);
private:
	torch::jit::script::Module module_;
	torch::Device device_;
};
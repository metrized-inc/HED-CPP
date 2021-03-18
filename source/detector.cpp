#include "detector.h"

Detector::Detector(const char* model_path, const torch::DeviceType& device_type){
    model = HED(model_path, device_type);
}


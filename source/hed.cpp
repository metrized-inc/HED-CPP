#include "hed.h"

HED::HED(const char* model_path, const torch::DeviceType& device_type) : device_(device_type) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    module_.to(device_);
    module_.eval();
}

void HED::Run(const cv::Mat& img, torch::Tensor& output) {
    torch::NoGradGuard no_grad;
    // preprocess image
    cv::Mat img_input = img.clone();
    cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);
    img_input.convertTo(img_input, CV_32FC3);
    cv::Scalar statistics = cv::Scalar(122.67891434,
                                    116.66876762,
                                    104.00698793);
    subtract(img_input, statistics, img_input); // minus statistics

    auto tensor_img = torch::from_blob(img_input.data, { 1, img_input.rows, img_input.cols, img_input.channels() }).to(device_);

    tensor_img = tensor_img.permute({ 0, 3, 1, 2 }).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);

    // inference
    torch::jit::IValue out = module_.forward(inputs);
    output = out.toTuple()->elements()[5].toTensor();

}

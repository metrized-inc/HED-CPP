#include "hed.h"

using namespace std;
using namespace cv;

int main(int argc, const char* argv[]) {
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    }
    else {
        device_type = torch::kCPU;
    }
    if (argc != 3) {
        cerr << "usage: test <path-to-exported-script-module> <path-to-image>" << endl;
        return -1;
    }
    HED model = HED(argv[1], device_type);


    // load image
    Mat image = imread(argv[2]);

    Mat temp_img = Mat::zeros(image.rows, image.cols, CV_32FC3);

    // run once on empty image to warmup
    torch::Tensor output;
    model.Run(temp_img, output);

    model.Run(image, output);

    cout << "Ran through the model" << endl;
    // change tensor to opencv mat
    output = output.permute({ 0, 2, 3, 1 });
    output = output.squeeze(0).detach();
    output = output.mul(255).clamp(0, 255).to(torch::kU8);
    output = output.to(torch::kCPU);
    cv::Mat resultImg(480, 480, 0);
    std::memcpy((void*)resultImg.data, output.data_ptr(), sizeof(torch::kU8) * output.numel());
    imshow("image", resultImg);
    waitKey(0);


    
}

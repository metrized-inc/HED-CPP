# hed-cpp

## Dependencies
- Libtorch 1.7.1
- OpenCV 4.5.1
- CMake 3.0 or higher
- Microsoft Visual Studio 2019

## Setup
```
git clone https://github.com/michelle-aubin/hed-cpp.git
cd hed-cpp
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```
Change ```/path/to/libtorch``` to the full path to your Libtorch installation. If you have not installed Libtorch you can do so [here](https://pytorch.org/).

## Running the application
```
cd Release
.\test-hed.exe [torchscript model] [image file]
```
```[torchscript model]``` is the .pt file of the torchscript model and ```[image file]``` is the filename of the image you want to run through the model

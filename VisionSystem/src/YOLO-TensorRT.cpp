// About
/*
* Detects objects with a YOLO object detection neural network.
* 
* Used by ObjectDetection.cpp.
* 
* The TensorRT is only optimized for a Jetson Nano. This is TensorRT inference is therefor only supposed to 
* work on a Jetson Nano, and not on Windows. Run the Object Detection with Darknet on Windows instead.
* 
* Load and setup the network with BuildNetwork().
* Run interference (detect objects) with Detect().
* Free memory before exit with Destroy().
*/

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

#include "yolo_layer.h"
#include "YOLO-TensorRT.h"

using std::cout, std::endl;

namespace yolo_tensorrt {

    // ######## ######## ######## ######## Parameters ######## ######## ######## ########
#pragma region Parameters

    const int BATCH_SIZE = 1;
    const int DLA_CORE = -1;

    const float IOU_THRESHOLD = 0.50;
    float confidenceThreshold;

    int imageWidth;
    int imageHeight;

    int inputChannels;
    int inputWidth;
    int inputHeight;
    int nClasses;

    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::vector<void*> buffers;
    std::vector<nvinfer1::Dims> input_dims; // expect only one input
    std::vector<nvinfer1::Dims> output_dims; // but two outputs

    float* cpu_nput;
    int cpu_input_size;
    float* cpu_output1;
    int cpu_output1_size;
    float* cpu_output2;
    int cpu_output2_size;

    cv::Mat rgbFrame;
    cv::Size input_size; 
    cv::cuda::GpuMat gpu_frame;
    cv::cuda::GpuMat resized;
    cv::cuda::GpuMat convertedImage;
    std::vector<cv::cuda::GpuMat> chw;

#pragma endregion Parameters


    // ######## ######## ######## ######## Utilities ######## ######## ######## ########
#pragma region Utilities

    // Class to log errors, warnings, and other information during the build and inference phases
    class Logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) override {
            // remove this 'if' if you need more logged info
            if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
                std::cout << msg << "\n";
            }
        }
    } gLogger;

    // Destroy TensorRT objects if something goes wrong
    struct TRTDestroy
    {
        template <class T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };

    // Pointer
    template <class T>
    using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

    // Calculate size of tensor
    size_t getSizeByDim(const nvinfer1::Dims& dims)
    {
        size_t size = 1;
        for (size_t i = 0; i < dims.nbDims; ++i)
        {
            size *= dims.d[i];
        }
        return size;
    }

#pragma endregion Utilities


    // ######## ######## ######## ######## Preprocessing Stage ######## ######## ######## ########
#pragma region Preprocessing Stage

    // Preprocess image before detection
    void preprocessImage(cv::Mat frame, float* gpu_input)
    {
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        
        // upload image to GPU
        gpu_frame.upload(rgbFrame);

        // resize
        cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

        // normalize
        resized.convertTo(convertedImage, CV_32FC3, 1.f / 255.f);
        
        // Split channel wise
        cv::cuda::split(convertedImage, chw);
    }

#pragma endregion Preprocessing Stage


    // ######## ######## ######## ######## Post Processing Stage ######## ######## ######## ########
#pragma region Post Processing Stage

    bool Compare (BoundingBox i, BoundingBox j) {
        return i.confidence > j.confidence;
    }
    
    // Remove bounding boxes that overlaps
    void NonMaximumSuppressionBounginBoxSelection(std::vector<BoundingBox>& detections) {
        std::sort(detections.begin(), detections.end(), Compare); // sort bounding boxes on the confidence score, in falling order
        
        // Loop over all boxes (index)
        int index = 0;
        while (index < detections.size()) {
            // Loop over all boxes i with lower confidence score than index
            int i = index + 1;
            int area = detections[index].w * detections[index].h;
            while (i < detections.size()) {
                int areaCompared = detections[i].w * detections[i].h;

                // Calculate the intersection area between i and index
                int xx1 = std::max(detections[index].x, detections[i].x);
                int yy1 = std::max(detections[index].y, detections[i].y);
                int xx2 = std::min(detections[index].x + detections[index].w, detections[i].x + detections[i].w);
                int yy2 = std::min(detections[index].y + detections[index].h, detections[i].y + detections[i].h);

                int intersectionWidth = std::max(0, xx2 - xx1 + 1);
                int intersectionHeight = std::max(0, yy2 - yy1 + 1);
                int intersection = intersectionWidth * intersectionHeight;
                int unionCompared = area + areaCompared - intersection;
                float IOU = intersection / (float)unionCompared;

                // Remove bounding box i if it has a high overlap with index
                if (IOU > IOU_THRESHOLD) {
                    detections.erase(detections.begin() + i);
                }
                else {
                    i++;
                }
            }
            index++;
        }
    }

    // Filter the bounding boxes from low confidences and high overlapping.
    // Converts bounding boxes to pixel coordinates and crops the boxes to fit into the image. 
    std::vector<BoundingBox> postprocessResults(float* gpu_output1, const nvinfer1::Dims& dims1, float* gpu_output2, const nvinfer1::Dims& dims2)
    {
        // Filter low confidence detections and combine the output layers and convert x, y, w, h from [0,1] to pixel coordinates
        std::vector<BoundingBox> detections;
        for (int i = 0; i < cpu_output1_size; i += 7) {
            if (cpu_output1[i + 4] * cpu_output1[i + 6] > confidenceThreshold) {
                detections.push_back(BoundingBox(cpu_output1[i] * imageWidth, cpu_output1[i + 1] * imageHeight, cpu_output1[i + 2] * imageWidth, cpu_output1[i + 3] * imageHeight,
                    cpu_output1[i + 4], cpu_output1[i + 5], cpu_output1[i + 6]));
            }
        }
        for (int i = 0; i < cpu_output2_size; i += 7) {
            if (cpu_output2[i + 4] * cpu_output2[i + 6] > confidenceThreshold) {
                detections.push_back(BoundingBox(cpu_output2[i] * imageWidth, cpu_output2[i + 1] * imageHeight, cpu_output2[i + 2] * imageWidth, cpu_output2[i + 3] * imageHeight,
                    cpu_output2[i + 4], cpu_output2[i + 5], cpu_output2[i + 6]));
            }
        }

        // Non Maximum Suppression: remove overlapping bounding boxes
        std::vector<BoundingBox> classSpecificDetections;
        std::vector<BoundingBox> resultVector;
        for (int iClass = 0; iClass < nClasses; iClass++) {
            classSpecificDetections.clear();
            for (int i = 0; i < detections.size(); i++) {
                if (detections[i].classId == iClass) {
                    classSpecificDetections.push_back(detections[i]);
                }
            }

            NonMaximumSuppressionBounginBoxSelection(classSpecificDetections);

            for (int i = 0; i < classSpecificDetections.size(); i++)
                resultVector.push_back(classSpecificDetections[i]);
        }

        // Crop the bounding boxes to fit into the image
        for (int i = 0; i < resultVector.size(); i++) {
            int newX = std::max(0, std::min(imageWidth - 1, resultVector[i].x));
            int newY = std::max(0, std::min(imageHeight - 1, resultVector[i].y));
            int diffX = std::abs(resultVector[i].x - newX);
            int diffY = std::abs(resultVector[i].y - newY);
            int newW = std::min(resultVector[i].w - diffX, imageWidth - newX - 1);
            int newH = std::min(resultVector[i].h - diffY, imageHeight - newY - 1);
            resultVector[i].x = newX;
            resultVector[i].y = newY;
            resultVector[i].w = newW;
            resultVector[i].h = newH;
        }
        return resultVector;
    }

#pragma endregion Post Processing Stage


    // ######## ######## ######## ######## Setup Network ######## ######## ######## ########
#pragma region Setup Network

    // Load YOLO TensorRT engine from file
    nvinfer1::ICudaEngine* loadEngine(const std::string& engine)
    {
        // Read data
        std::ifstream engineFile(engine, std::ios::binary);
        if (!engineFile)
        {
            cout << "[ERROR] opening engine file: " << engine << std::endl;
            return nullptr;
        }

        engineFile.seekg(0, engineFile.end);
        long int fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);

        void* engineData = malloc(fsize);
        engineFile.read((char*)engineData, fsize);
        if (!engineFile)
        {
            cout << "[ERROR] loading engine file: " << engine << std::endl;
            return nullptr;
        }

        // Create logger
        TRTUniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(gLogger) };

        // Read YOLO plug-in
        nvinfer1::YoloPluginCreator pluginCreator();
        float tmp = 1.0;
        nvinfer1::YoloLayerPlugin plugin(1, 1, 1, &tmp, 1, 1, 1, 1.0, 1); // Dummy, only to import the plug-in. 
        cout << "[OK] Imported YOLO plug-in: " << plugin.getPluginType() << endl;

        initLibNvInferPlugins(&gLogger, "nvinfer1");

        return runtime->deserializeCudaEngine(engineData, fsize, nullptr);
    }

    // Build the YOLO Network and allocate memory for the network, input, output and intermediate data
    void BuildNetwork(std::string enginePath, int _nClasses, float _confidenceThreshold, int _imageWidth, int _imageHeight) {
        // Load engine
        nClasses = _nClasses;
        confidenceThreshold = _confidenceThreshold;
        imageWidth = _imageWidth;
        imageHeight = _imageHeight;
        engine = loadEngine(enginePath);
        if (!engine) {
            cout << "[ERROR] Engine creation failed" << endl;
            exit(-1);
        }
        cout << "[OK] Loaded YOLO TensorRT Engine" << endl;

        // Create some space to store intermediate activation values (the data between input and output)
        context = engine->createExecutionContext();

        // Get sizes of input and output and allocate memory required for input data and for output data
        buffers = std::vector<void*>(engine->getNbBindings());
        for (size_t i = 0; i < engine->getNbBindings(); ++i)
        {
            auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * BATCH_SIZE * sizeof(float);
            cudaMalloc(&buffers[i], binding_size);
            if (engine->bindingIsInput(i))
            {
                input_dims.emplace_back(engine->getBindingDimensions(i));
                auto inputDims = engine->getBindingDimensions(i).d;
                inputChannels = inputDims[1];
                inputHeight = inputDims[2];
                inputWidth = inputDims[3];
                cout << "Building network with input size: " << inputChannels << " x " << inputHeight << " x " << inputWidth << endl;
            }
            else
            {
                output_dims.emplace_back(engine->getBindingDimensions(i));
            }
        }
        if (input_dims.empty() || output_dims.empty())
        {
            std::cerr << "[ERROR] Expect at least one input and one output for network\n";
            exit(-1);
        }

        // Allocate memory for the output
        cudaHostAlloc((void**) &cpu_output1, getSizeByDim(output_dims[0]) * BATCH_SIZE * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc((void**) &cpu_output2, getSizeByDim(output_dims[1]) * BATCH_SIZE * sizeof(float), cudaHostAllocMapped);

        buffers[1] = cpu_output1;
        buffers[2] = cpu_output2;

        cpu_output1_size = getSizeByDim(output_dims[0]) * BATCH_SIZE;
        cpu_output2_size = getSizeByDim(output_dims[1]) * BATCH_SIZE;

        // Allocate memory for the input
        rgbFrame = cv::Mat(cv::Size(imageWidth, imageHeight), CV_8UC3);
        input_size = cv::Size(inputWidth, inputHeight);
        gpu_frame = cv::cuda::GpuMat(cv::Size(imageWidth, imageHeight), CV_8UC3);
        resized = cv::cuda::GpuMat(input_size, CV_8UC3);
        convertedImage = cv::cuda::GpuMat(input_size, CV_32FC3);
        for (size_t i = 0; i < inputChannels; ++i)
        {
            chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, (float*)buffers[0] + i * inputWidth * inputHeight));
        }

        cout << "[SUCCESS] Building Network Finished" << endl;
    }

    // Free allocated data
    void Destroy() {
        cudaFree(cpu_output1);
        cudaFree(cpu_output2);
        for (void* buf : buffers)
        {
            cudaFree(buf);
        }
        cout << "Network destroyed" << endl;
    }

#pragma endregion Setup Network


    // ######## ######## ######## ######## Run Inference ######## ######## ######## ########
#pragma region Run Inference

    // Detect objects from an image with YOLOv4-tiny.
    // Returns a result vector with bounding boxes.
    std::vector<BoundingBox> Detect(cv::Mat frame) {
        // Preprocess input data
        preprocessImage(frame, (float*)buffers[0]);

        // Detect
        context->execute(BATCH_SIZE, buffers.data());

        // Post process results
        std::vector<BoundingBox> resultVector = postprocessResults((float*)buffers[1], output_dims[0], (float*)buffers[2], output_dims[1]);
        return resultVector;
    }

#pragma endregion Run Inference

}
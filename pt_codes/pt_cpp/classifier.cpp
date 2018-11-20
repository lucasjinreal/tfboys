//
// Created by jintain on 11/20/18.
//
/**
 * A simple pytorch classifier inference code test
 *
 * this code will load trained model with python convert
 * it into c++ need model, then load it and inference
 * just for going through all the process deploying pytorch
 * model in C++ production environment
 */



#include "torch/script.h"
#include <torch/script.h>
#include <torch/torch.h>
//#include <torch/Tensor.h>
#include <ATen/Tensor.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>
#include <memory>
//#include <vec>


using namespace std;


void load_labels(string label_f, vector<string> labels) {
    ifstream ins(label_f);
    string line;
    while (getline(ins, line)) {
        labels.push_back(line);
    }
}


int main(int argc, const char* argv[]) {

    if (argc != 4) {
        cout << "ptcpp path/to/scripts/model.pt path/to/image.jpg path/to/label.txt\n";
        return -1;
    }

    shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

    if (module == nullptr) {
        cerr << "model load error from " << argv[1] << endl;
    }
    cout << "Model load ok.\n";

    // load image and transform
    cv::Mat image;
    image = cv::imread(argv[2], 1);
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0/255);
    cv::resize(img_float, img_float, cv::Size(224, 224));
    auto img_tensor = torch::CUDA(torch::kFloat32).tensorFromBlob(img_float.data, {1, 224, 224, 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);
    auto img_var = torch::autograd::make_variable(img_tensor, false);

    vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var);
    torch::Tensor out_tensor = module->forward(inputs).toTensor();
    cout << out_tensor.slice(1, 0, 10) << '\n';

    // load label
    vector<string> labels;
    load_labels(argv[3], labels);
    cout << "Found all " << labels.size() << " labels.\n";

    // out tensor sort, print the first 2 category
    std::tuple<torch::Tensor,torch::Tensor> result = out_tensor.sort(-1, true);
    torch::Tensor top_scores = std::get<0>(result)[0];
    torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);

    auto top_scores_a = top_scores.accessor<float,1>();
    auto top_idxs_a = top_idxs.accessor<int,1>();

    for (int i = 0; i < 5; ++i) {
        int idx = top_idxs_a[i];
        std::cout << "top-" << i+1 << " label: ";
        std::cout << labels[idx] << ", score: " << top_scores_a[i] << std::endl;
    }

    return 0;
}
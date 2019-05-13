#include "torch/script.h"
#include <torch/script.h>
#include <torch/torch.h>
//#include <torch/Tensor.h>
#include <ATen/Tensor.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>


using namespace std;

int main() {

    torch::Tensor foo = torch::rand({12, 12});
    cout << "foo: " << foo << endl;

    cout << "foo size: " << foo.size(0) << "x" << foo.size(1) << endl;

    for (int i=0; i<4; i++) {
        cout << foo[0][i] << " ";
    }
}
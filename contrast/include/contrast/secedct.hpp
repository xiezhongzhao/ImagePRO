// MIT License
// Copyright (c) 2022 - xiezhongzhao
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// @Author:  xiezhongzhao
// @Email:   2234309583@qq.com
// @Data:    2023/11/16 19:37
// @Version: 1.0

#ifndef IMAGEPRO_SECEDCT_HPP
#define IMAGEPRO_SECEDCT_HPP

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <map>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>


using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::unique_ptr;
using std::max;
using std::min;
using std::ifstream;
using std::ios;
using std::strcmp;
using std::move;

using cv::Mat;
using cv::Point2d;
using cv::Point;
using cv::Scalar;
using cv::format;
using cv::FONT_HERSHEY_SIMPLEX;
using cv::Size;
using cv::INTER_AREA;
using cv::BORDER_CONSTANT;

namespace contrast{

    void SECE(cv::Mat& y);

}
#endif //IMAGEPRO_SECEDCT_HPP

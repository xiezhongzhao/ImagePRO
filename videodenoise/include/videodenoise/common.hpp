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
// @Data:    2023/11/16 11:23
// @Version: 1.0

#ifndef VIDEODENOISER_COMMON_HPP
#define VIDEODENOISER_COMMON_HPP

#pragma once

#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <io.h>

#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utility.hpp>

#include <timer/timer.hpp>

#ifdef ORIG
#define original
#elif defined(ARM_NEON)
#include <arm_neon.h>
#elif defined(SSE)
#include <xmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#endif

#define LOG(x) std::cerr
const double PI = 3.141592653589793238463;

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

#endif //VIDEODENOISER_COMMON_HPP

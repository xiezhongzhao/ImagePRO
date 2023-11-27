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
// @Data:    2023/11/16 19:23
// @Version: 1.0

#include <timer/timer.hpp>
namespace Timer{

    Timer::Timer(std::string timer_name) {
        d_start = std::chrono::high_resolution_clock::now();
        _timer_name = timer_name;
    }

    Timer::Timer(std::string timer_name, double cmp_ms) {
        d_start = std::chrono::high_resolution_clock::now();
        _timer_name = timer_name;
        _cmp_ms = cmp_ms;
    }

    Timer::~Timer() {}

    void Timer::stop() {

        auto d_end = std::chrono::high_resolution_clock::now();
        auto _start =
                std::chrono::time_point_cast<std::chrono::microseconds>(d_start)
                        .time_since_epoch().count();
        auto _end =
                std::chrono::time_point_cast<std::chrono::microseconds>(d_end)
                        .time_since_epoch().count();
        auto duration = _end - _start;
        _ms = duration * 0.001;

        std::cout << _timer_name << " time cost: " << _ms << "ms ";
        if (_cmp_ms != 0)
            std::cout << "speedup: " << _cmp_ms / _ms << "x";
        std::cout << "\n";
    }

    double Timer::get() {
        return _ms;
    }
}

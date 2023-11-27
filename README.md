[TOC]

## **视频降噪性能测试**

1. 测试平台

   * Inter(R) Core(TM) i-10700 CPU @ 2.9GHz 
   * 内存（RAM）: 16.0 GB
   * 系统：Windows 10 64位操作系统

2. 测试结果

  任务要求是在测试平台上，视频降噪帧率达到120FPS以上

  A: EstimateMotion  （DIS光流估计像素的相对移动）    
  B: GetYUVAbsoluteMotion  （计算像素的绝对位置信息）  
  C: RemapYUV  （映射到移动后的像素坐标）  
  D: Fusion   （前后帧进行融合）  
  E: FilterYUV   （对三个通道进行空域滤波）   

| 视频分辨率 | A (ms) | B (ms) | C (ms) | D (ms) | E (ms) | Total (ms) |
| :--------: | :----: | :----: | :----: | :----: | ------ | :--------: |
|    1080    |  16.5  |  3.6   |  7.65  |  12.4  | 1.46   |     46     |

ToDoList:

- [ ] 将B模块优化到1ms以内
- [ ] 将D模块优化到2ms以内
- [ ] 将C模块优化到2ms以内
- [ ] 将DIS光流估计的算法复杂度降低4倍
- [ ] 将A模块优化到5ms以内

## 视频降噪改进思路

视频降噪的主要性能瓶颈在于运动估计和帧间融合这两块，运动估计需要改进光流跟追算法，在保证效果的同时，降低算法复杂度；帧间融合的耗时取决于融合的帧数，帧数越多去噪效果越好，但是耗时越长，因此需要做权衡。帧间融合的优化完全可以通过工程方式降低耗时。

1. 运动估计的优化思路（Google Pixel 2）

   在KLT金字塔稀疏光流追踪过程中，在更粗的金字塔层的整数精度的运动矢量足够精确来初始化运动矢量搜索到下一个更细的层，因此，除了最后一层外，可以去除所有不必要的插值来计算亚像素对齐误差。最后，为了增加运动搜索范围，减少迭代次数，使用最粗层的水平和垂直的一维投影，通过互相关来估计全局运动。这种优化方法时间比原来缩短4倍。（具体内容参见文献[4]）

2. 帧间融合的优化思路 （工程优化指令集、多线程、OpenCL、OpenGL等）

   帧间融合可以通过工程优化大幅度降低耗时。

___



## **参考文献**

1. MeshFlow: Minimum Latency Online Video Stabilization
2. SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization
3. MESHFLOW VIDEO DENOISING
4. Real-Time Video Denoising On Mobile Phones


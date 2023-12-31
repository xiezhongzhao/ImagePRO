## **视频降噪性能测试**

|        原始视频帧/降噪后视频帧        |
|:--------------------------:|
|   ![](./doc/raw_img.png)   |
| ![](./doc/img_denoise.png) |

1. 测试平台

   * Inter(R) Core(TM) i-10700 CPU @ 2.9GHz 
   * 内存（RAM）: 16.0 GB
   * 系统：Windows 10 64位操作系统

2. 测试结果

  任务要求是在测试平台上，视频降噪（分辨率1080P）帧率达到120FPS以上

  A: EstimateMotion  （DIS光流估计像素的相对移动）    
  B: GetYUVAbsoluteMotion  （计算像素的绝对位置信息）  
  C: RemapYUV  （映射到移动后的像素坐标）  
  D: Fusion   （前后帧进行融合）  
  E: FilterYUV   （对三个通道进行空域滤波）   

|  版本/阶段  | A (ms)  | B (ms) | C (ms) | D (ms) | E (ms) |   FPS    | Total (ms) |
| :---------: | :-----: | :----: | :----: | :----: | :----: | :------: | :--------: |
| version1.0  |  16.5   |  3.6   |  7.65  |  12.4  |  1.46  |   21.7   |     46     |
| version 2.0 |   4.9   |  0.87  |  2.0   |  7.5   |  0.51  |   63.4   |   15.78    |
| version 2.1 |   5.0   |  1.0   |  2.0   |   10   |  0.25  |   54.8   |   18.25    |
| version 2.2 |   5.0   |  0.9   |  2.0   |  6.7   |  0.25  |   67.3   |   14.85    |
| version 2.3 | **2.6** |  0.9   |  2.0   |  6.7   |  0.25  | **80.3** | **12.45**  |
| version 3.0 |    *    |   *    |   *    |   *    |   *    |   120    |    8.2     |

ToDoList:

2023.12.13 v.2.0

（1）在保证效果的情况下，将原始YUV**图片缩小两倍**；

（2）对D模块进行速度优化，提前计算前后帧的权重存入数组；

- [x] 将B模块优化到1ms以内
- [x] 将C模块优化到2ms以内
- [x] 将A模块优化到5ms以内

2023.12.15 v.2.1

（1）改用新的策略进行前后帧融合，效果稳定，但是**耗时增加**（10ms左右）；

- [x] 优化D模块的**融合效果**

2023.12.19 v.2.2

（1）对D模块进行速度优化，采用SSE和parallel_for进行加速，但是**加速效果不稳定**；     

（2）后续进一步速度优化，需要采用**汇编加速**；

- [x] 将D模块耗时在**6.5ms**到7.5ms之间

2023.12.23 v2.3

（1）改进DIS算法源码，将A模块优化到**2.6ms**以内；

- [x] 将A模块优化到**2.6ms**以内

2023.12.30 v3.0

（1）从原理上将DIS算法复杂度降低4倍，将A模块优化到2ms以内；

（2）采用OpenGL将D模块耗时优化到2ms以内；

- [ ] 将DIS光流估计的算法复杂度降低4倍
- [ ] 将D模块优化到2ms以内

## 视频降噪改进思路

视频降噪的主要性能瓶颈在于运动估计和帧间融合这两块，运动估计需要改进光流跟追算法，在保证效果的同时，降低算法复杂度；帧间融合的耗时取决于融合的帧数，帧数越多去噪效果越好，但是耗时越长，因此需要做权衡。帧间融合的优化完全可以通过工程方式降低耗时。

1. 运动估计的优化思路（Google Pixel 2）

   在KLT金字塔稀疏光流追踪过程中，在更粗的金字塔层的整数精度的运动矢量足够精确来初始化运动矢量搜索到下一个更细的层，因此，除了最后一层外，可以去除所有不必要的插值来计算亚像素对齐误差。最后，为了增加运动搜索范围，减少迭代次数，使用最粗层的水平和垂直的一维投影，通过互相关来估计全局运动。这种优化方法时间比原来缩短4倍。（具体内容参见文献[4]）

2. 帧间融合的优化思路 （工程优化指令集、多线程、OpenCL、OpenGL等）

   帧间融合可以通过工程优化大幅度降低耗时。

## **参考文献**

1. MeshFlow: Minimum Latency Online Video Stabilization
2. SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization
3. MESHFLOW VIDEO DENOISING
4. Real-Time Video Denoising On Mobile Phones


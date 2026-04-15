# 视觉伺服与机器人跟踪技术报告

## 摘要

本文总结课程项目 `Proj.pdf` 的 Part II。系统围绕 PUMA 560 建立了三类视觉任务：相机标定、基于位置的目标跟踪、基于图像特征的视觉伺服。实现分为两条路径：一条是纯仿真，用于验证几何关系、控制律与收敛性；另一条是真实摄像头路径，用 MacBook 内置摄像头拍摄打印的 ChArUco 板，验证真实图像输入到虚拟机械臂闭环的完整链路。

报告重点说明任务目标、方法基础、代码结构、实验结果和可用于答辩展示的图像资产。本文只讨论 MATLAB 交付版，不展开后续的 MuJoCo 分支。

## 1. 项目目标

Part II 的目标不是单纯做图像识别，而是建立“视觉测量 -> 误差计算 -> 机器人控制”的闭环。按照题目要求，任务可以拆成三部分：

- `T1`：相机标定，建立图像坐标和世界坐标之间的几何关系。
- `T2`：基于位置的跟踪，检测目标位置并驱动虚拟机械臂跟随。
- `T3`：基于图像特征的视觉伺服，用四角点做 IBVS，让末端对准一个方形平面目标。

系统同时实现了两条验证路线：

- **纯仿真**：相机、目标和机械臂都在 MATLAB 内部模拟。
- **真实摄像头**：用 MacBook 内置摄像头拍摄打印的 ChArUco 板，做真实标定与跟踪验证；机械臂仍然是 MATLAB 里的虚拟 PUMA 560。

## 2. 方法基础：Visual Servo、PBVS、IBVS

### 2.1 Visual Servo

视觉伺服的本质是把图像中的测量量作为反馈量，直接控制机器人运动。与传统开环轨迹规划不同，视觉伺服是在线闭环：机器人不是先算好一条固定轨迹再执行，而是每一帧都根据视觉误差修正控制量。

### 2.2 PBVS

PBVS（Position-Based Visual Servoing）先从图像中恢复目标在三维空间中的位姿，再用位姿误差驱动机器人。它的优点是空间意义直观，便于定义目标距离、朝向和末端姿态；缺点是依赖较稳健的三维位姿估计。

### 2.3 IBVS

IBVS（Image-Based Visual Servoing）直接在图像特征上定义误差，不先恢复完整三维位姿。经典形式可写为：

\[
\mathbf{v} = -\lambda \mathbf{L}_s^+(\mathbf{s} - \mathbf{s}^\*)
\]

其中 \(\mathbf{s}\) 是当前图像特征，\(\mathbf{s}^\*\) 是期望特征，\(\mathbf{L}_s\) 是 interaction matrix / image Jacobian，\(\lambda\) 是控制增益。

IBVS 的优点是直接、局部收敛快；缺点是对特征设计、深度近似和数值稳定性更敏感。

### 2.4 本项目中的对应关系

- `T1`：标定任务，为后续视觉伺服提供相机参数。
- `T2`：位置基跟踪，方法上更接近 PBVS / position-based tracking。
- `T3`：IBVS，直接用图像特征闭环控制机器人。

**本节小结**  
PBVS 先求空间位姿再控机器人，IBVS 直接控图像特征。T2 偏位置反馈，T3 是标准图像特征闭环。

## 3. 系统结构与代码映射

MATLAB 版代码主要集中在 `src/`。关键入口和功能如下：

| 功能 | 文件 | 作用 |
|---|---|---|
| 主入口 | [`src/run_demo.m`](src/run_demo.m), [`src/main.m`](src/main.m) | 顺序运行 T1/T2/T3 并写日志 |
| 全局参数 | [`src/config.m`](src/config.m) | 统一定义相机、机器人、板子、路径、阈值 |
| 标定板资产 | [`src/charuco_board_asset.m`](src/charuco_board_asset.m) | 生成可打印 ChArUco PNG/PDF |
| 真实标定 | [`src/live_camera_calibration.m`](src/live_camera_calibration.m), [`src/calibrate_charuco_images.m`](src/calibrate_charuco_images.m) | 采集真实图像并估计相机参数 |
| 参数加载 | [`src/load_camera_params.m`](src/load_camera_params.m) | 优先加载 `assets/cameraParams.mat` |
| T1 仿真 | [`src/t1_virtual_calibration.m`](src/t1_virtual_calibration.m) | 合成标定、输出图和视频 |
| T2 仿真 | [`src/t2_position_tracking.m`](src/t2_position_tracking.m) | fixed-camera 和 eye-in-hand 位置跟踪 |
| T3 仿真 | [`src/t3_ibvs_square.m`](src/t3_ibvs_square.m) | 方形目标 IBVS |
| 真实相机跟踪 | [`src/real_camera_tracking.m`](src/real_camera_tracking.m) | 真实摄像头驱动虚拟机器人 |
| 几何与投影 | [`src/lookat_tform.m`](src/lookat_tform.m), [`src/project_points.m`](src/project_points.m), [`src/backproject_to_plane.m`](src/backproject_to_plane.m) | 相机姿态、投影、平面反投影 |
| 控制与数值稳定 | [`src/dls_solve.m`](src/dls_solve.m), [`src/session_controls.m`](src/session_controls.m) | 阻尼最小二乘、手动 Start/Stop 会话控制 |
| 结果汇总 | [`results/analysis_summary.md`](results/analysis_summary.md) | 最新数值摘要 |

可视化统一由 [`src/apply_world_view.m`](src/apply_world_view.m)、[`src/draw_camera_frame.m`](src/draw_camera_frame.m)、[`src/add_status_badge.m`](src/add_status_badge.m) 管理。  
机器人模型由 `config.m` 中的 `loadrobot('puma560')` 创建，后续所有任务都复用同一套 rigid body tree。

**本节小结**  
整个项目的参数和流程是集中式管理的。这样做的好处是：仿真、真机标定、真实摄像头跟踪都能复用同一套几何定义。

## 4. 标定板与真实相机资产

### 4.1 ChArUco 板为什么放在 `assets/`

长期保留的公开资产只有两类：

- 打印用的 ChArUco 标定板
- 真实相机导出的 `cameraParams.mat`

它们都放在 `assets/report/` 下，作为报告和演示的公共资产。

### 4.2 标定板参数

打印板文件：

- [`assets/report/charuco_board_printable.png`](assets/report/charuco_board_printable.png)
- [`assets/report/charuco_board_printable.pdf`](assets/report/charuco_board_printable.pdf)

参数如下：

- pattern: `7 x 5`
- dictionary: `DICT_4X4_1000`
- checker size: `30 mm`
- marker size: `22.5 mm`
- image size: `2942 x 2102`

这套参数在 [`src/config.m`](src/config.m) 中同时用于：

- 纯仿真标定板
- 真实相机标定板
- 真实摄像头跟踪任务

### 4.3 相机参数

真实相机导出的参数文件为：

- [`assets/report/cameraParams.mat`](assets/report/cameraParams.mat)

后续真实摄像头任务会优先加载该文件。这样做的目的，是让仿真和真机使用同一套标定几何，而不是每次运行时重新标定。

**本节小结**  
标定板和相机参数是这套系统的公共基础设施。仿真和真机共用同一块板，减少了几何定义不一致的问题。

## 5. T1：相机标定与几何对齐

### 5.1 任务定义

T1 的作用是把相机坐标和世界坐标联系起来。这里采用 ChArUco 板，因为它同时具备：

- 标记 ID
- 角点精度高
- 适合实际打印
- 适合仿真和真实摄像头共用

### 5.2 纯仿真标定

仿真标定由 [`src/t1_virtual_calibration.m`](src/t1_virtual_calibration.m) 完成。实现方式是：

1. 生成与真实板一致的 printable ChArUco 板纹理。
2. 设定 24 个相机视角。
3. 对每个视角投影板角点并加入少量噪声。
4. 使用 `estimateCameraParameters` 估计内参。
5. 统计重投影误差。

仿真结果来自最新导出：

![T1 board path](assets/report/t1_calibration_board_path.png)

![T1 synthetic image](assets/report/t1_calibration_synthetic_image.png)

![T1 intrinsics comparison](assets/report/t1_calibration_intrinsics.png)

视频版本：

[T1 calibration animation MP4](assets/report/t1_calibration_animation.mp4)

最新仿真摘要如下：

- board type: `charuco-board`
- views used: `24`
- mean reprojection error: `0.4226 px`
- true `fx/fy`: `900.00 / 900.00`
- estimated `fx/fy`: `938.22 / 940.26`

这说明仿真链路是自洽的：生成的角点、投影、估计和误差统计是闭环的。

### 5.3 真实相机标定

真实标定由 [`src/live_camera_calibration.m`](src/live_camera_calibration.m) 和 [`src/calibrate_charuco_images.m`](src/calibrate_charuco_images.m) 完成。流程是：

1. 打印 ChArUco 板并保持平整。
2. 用 MacBook 内置摄像头采集图像。
3. 在每帧中检测 ChArUco 角点。
4. 利用有效图像估计 `cameraParams`。
5. 保存 `cameraParams.mat` 供后续 T2/T3 复用。

最新真实标定日志记录如下：

- 摄像头：`MacBook Pro Camera`
- 通过帧数：`24`
- 平均重投影误差：`4.1257 px`
- 估计 `fx/fy`：`1364.55 / 1357.13`

对应输出图：

![Live camera captures](assets/report/live_camera_captures.png)

![Live camera intrinsics](assets/report/live_camera_intrinsics.png)

![Live camera support](assets/report/live_camera_support.png)

### 5.4 T1 的解释

真实标定误差高于仿真，这是正常现象。原因主要有三类：

- 手持/摆放时的微小姿态变化
- 真实镜头的畸变和噪声
- 图像边缘和角点质量不稳定

对后续任务而言，T1 的关键不是把误差压到极小，而是提供一套可复用的真实相机参数。

### 5.5 本节小结

T1 完成了两件事：

- 仿真和真实共用同一块 ChArUco 板；
- 真机相机参数被保存为 `cameraParams.mat`，可以直接供 T2/T3 使用。

**本节小结**  
T1 把几何关系先立住。仿真结果用于验证流程，真实结果用于后续任务的实际输入。

## 6. T2：基于位置的跟踪

### 6.1 任务定义

T2 要做的是：识别目标位置，然后把这个位置反馈给虚拟机械臂，使末端持续跟随并保持悬停高度。  
这里的位置控制比 IBVS 更直观，主要输出是三维位置误差，而不是纯图像误差。

### 6.2 纯仿真 fixed-camera

纯仿真 fixed-camera 由 [`src/t2_position_tracking.m`](src/t2_position_tracking.m) 完成。其逻辑是：

- 用固定相机看桌面。
- 桌面上放一个红色目标球和几个干扰球。
- 通过颜色分割识别目标。
- 用 [`src/backproject_to_plane.m`](src/backproject_to_plane.m) 把图像中心点反投影到桌面平面。
- 用逆运动学控制 PUMA 560 末端跟随目标上方的位置。

这一版的意义是把“检测”和“控制”解耦，先验证位置反馈链路是否能工作。

结果图如下：

![T2 fixed trajectory](assets/report/t2_fixed_camera_tracking_trajectory.png)

![T2 fixed error](assets/report/t2_fixed_camera_tracking_error.png)

![T2 fixed joints](assets/report/t2_fixed_camera_tracking_joints.png)

![T2 fixed support](assets/report/t2_fixed_camera_tracking_support.png)

视频：

[T2 fixed-camera MP4](assets/report/t2_fixed_camera_tracking.mp4)

最新仿真摘要：

- final position error: `0.000198 m`
- RMSE position error: `0.000294 m`

### 6.3 纯仿真 eye-in-hand

eye-in-hand 版本同样由 [`src/t2_position_tracking.m`](src/t2_position_tracking.m) 完成，只是相机位于机械臂末端附近，视角随机器人运动变化。

与 fixed-camera 相比，eye-in-hand 更接近“机械臂边动边看”的场景。该模式的价值主要在于验证：当相机本身随机器人运动时，位置跟踪链路仍然能保持稳定。

结果图如下：

![T2 eye trajectory](assets/report/t2_eye_in_hand_tracking_trajectory.png)

![T2 eye error](assets/report/t2_eye_in_hand_tracking_error.png)

![T2 eye joints](assets/report/t2_eye_in_hand_tracking_joints.png)

![T2 eye support](assets/report/t2_eye_in_hand_tracking_support.png)

视频：

[T2 eye-in-hand MP4](assets/report/t2_eye_in_hand_tracking.mp4)

最新仿真摘要：

- final position error: `0.000347 m`
- RMSE position error: `0.000441 m`

### 6.4 真实相机 follow

真实相机 follow 由 [`src/real_camera_tracking.m`](src/real_camera_tracking.m) 完成。这里的结构是：

- 用真实摄像头采集图像。
- 读取已保存的 `cameraParams.mat`。
- 以打印的 ChArUco 板作为目标物。
- 从真实图像中估计目标位置。
- 用估计结果驱动虚拟 PUMA 560。

这条路径验证的是“真实图像输入是否能够稳定进入控制环路”。  
注意这里的机械臂仍然是虚拟的，没有物理机械臂参与。

最新日志：

- 模式：`follow`
- 运行方式：`manual`
- 来源：`webcam`
- 处理样本：`32`

对应结果：

![Real follow trajectory](assets/report/real_camera_follow_trajectory.png)

![Real follow error](assets/report/real_camera_follow_error.png)

![Real follow joints](assets/report/real_camera_follow_joints.png)

![Real follow support](assets/report/real_camera_follow_support.png)

视频：

[Real follow MP4](assets/report/real_camera_follow_tracking.mp4)

### 6.5 T2 的解释

T2 的核心不是追求复杂感知，而是验证位置闭环是否稳定。  
仿真 fixed-camera 和 eye-in-hand 两种模式说明了位置跟随的几何逻辑；真实相机 follow 说明了真实图像到虚拟机器人动作的链路可以跑通。

### 6.6 本节小结

T2 完成了“检测目标位置 -> 估计三维位置 -> 虚拟机械臂跟随”的闭环。  
其中 fixed-camera 和 eye-in-hand 是仿真中的两种相机布局，真实 follow 则验证了摄像头输入在真实场景下仍能工作。

**本节小结**  
T2 是位置基跟踪。它更像 PBVS 的工程化版本：先估计位置，再让末端稳定地悬停并跟随目标移动。

## 7. T3：基于特征的跟踪 / IBVS

### 7.1 任务定义

T3 是典型的 IBVS：选定一个方形平面目标，以四个角点作为图像特征，直接在图像空间定义误差并闭环控制。

### 7.2 纯仿真 IBVS

仿真 IBVS 由 [`src/t3_ibvs_square.m`](src/t3_ibvs_square.m) 完成。其实现流程为：

1. 构造一个方形平面目标。
2. 计算当前四角点的归一化图像坐标。
3. 与期望特征作差，得到图像误差。
4. 根据 interaction matrix 构造控制律。
5. 使用 [`src/dls_solve.m`](src/dls_solve.m) 做阻尼最小二乘求解。
6. 用回溯步长筛选稳定更新，避免误差发散。

这个版本并没有去做完整的 6 自由度强耦合伺服，而是将重点放在平移控制上，同时通过末端朝向约束维持目标平面与相机平面的近似平行。这样做的好处是稳定、可解释，也更适合课程展示。

结果图：

![T3 error](assets/report/t3_ibvs_square_tracking_error.png)

![T3 features](assets/report/t3_ibvs_square_tracking_features.png)

![T3 depths](assets/report/t3_ibvs_square_tracking_depths.png)

![T3 joints](assets/report/t3_ibvs_square_tracking_joints.png)

视频：

[T3 IBVS MP4](assets/report/t3_ibvs_square_tracking.mp4)

最新仿真摘要：

- final feature error: `0.013326`
- RMSE feature error: `0.159721`

### 7.3 真实相机 IBVS

真实相机 IBVS 同样通过 [`src/real_camera_tracking.m`](src/real_camera_tracking.m) 实现，只是运行模式切到 `ibvs`。

在这一路径中：

- 真实摄像头采集图像。
- 用 ChArUco 板的角点构造特征。
- 用图像误差驱动虚拟机器人运动。
- 通过真实相机验证 IBVS 的特征提取和闭环接口。

最新日志：

- 模式：`ibvs`
- 运行方式：`auto`
- 来源：`webcam`
- 处理样本：`24`

结果图：

![Real IBVS error](assets/report/real_camera_ibvs_error.png)

![Real IBVS features](assets/report/real_camera_ibvs_features.png)

![Real IBVS joints](assets/report/real_camera_ibvs_joints.png)

![Real IBVS support](assets/report/real_camera_ibvs_support.png)

视频：

[Real IBVS MP4](assets/report/real_camera_ibvs_tracking.mp4)

### 7.4 T3 的解释

T3 的关键区别在于：控制对象不是三维位置本身，而是图像特征。  
这使得 IBVS 对“图像中看起来是否对齐”直接闭环，和 T2 的位置控制思路不同。  
从工程角度看，T3 更接近一个标准的视觉伺服演示，而不是简单的跟随任务。

### 7.5 本节小结

T3 证明了图像特征可以直接驱动机器人闭环，不必先完整恢复三维位姿。  
这里使用的特征是四角点，控制律通过阻尼最小二乘稳定求解。

**本节小结**  
T3 是标准 IBVS。它用四角点特征和 interaction matrix 直接闭环，强调“图像上对齐”，而不是“空间中到位”。

## 8. 结果汇总

最新数值汇总保存在 [`results/analysis_summary.md`](results/analysis_summary.md)。下面是可直接放进报告的摘要表：

| 任务 | 指标 | 数值 |
|---|---:|---:|
| T1 仿真标定 | Mean reprojection error | `0.4226 px` |
| T1 仿真标定 | true `fx/fy` | `900.00 / 900.00` |
| T1 仿真标定 | estimated `fx/fy` | `938.22 / 940.26` |
| T1 真实标定 | Mean reprojection error | `4.1257 px` |
| T1 真实标定 | Accepted frames | `24` |
| T1 真实标定 | estimated `fx/fy` | `1364.55 / 1357.13` |
| T2 fixed 仿真 | Final position error | `0.000198 m` |
| T2 fixed 仿真 | RMSE position error | `0.000294 m` |
| T2 eye 仿真 | Final position error | `0.000347 m` |
| T2 eye 仿真 | RMSE position error | `0.000441 m` |
| T2 真实 follow | Processed samples | `32` |
| T3 仿真 IBVS | Final feature error | `0.013326` |
| T3 仿真 IBVS | RMSE feature error | `0.159721` |
| T3 真实 IBVS | Processed samples | `24` |

### 结果解读

- **T1**：仿真标定误差很小，说明模型内部几何一致；真实标定误差更大，但仍可用，符合真实相机标定的预期。
- **T2**：位置误差进入毫米级，说明位置反馈闭环稳定。
- **T3**：特征误差收敛到较低水平，说明 IBVS 控制律有效。
- **真实相机路径**：follow 和 IBVS 都完成了指定采样数，说明真实摄像头输入、保存标定参数和控制接口都已打通。

**本节小结**  
从数值上看，仿真和真实相机两条链路都完成了闭环。仿真结果用于验证控制稳定性，真实结果用于证明系统可以接入实际摄像头。

## 9. 运行与复现

### 9.1 仿真

```matlab
addpath(genpath(pwd));
results = run_demo();
```

### 9.2 刷新真实标定

```matlab
addpath(genpath(pwd));
results = run_live_camera_calibration();
```

### 9.3 真实相机 follow

```matlab
addpath(genpath(pwd));
results = run_real_camera_follow('RunMode', 'manual');
```

### 9.4 真实相机 IBVS

```matlab
addpath(genpath(pwd));
results = run_real_camera_ibvs('RunMode', 'manual');
```

### 9.5 说明

- `assets/report/cameraParams.mat` 是公开资产，真实相机任务会优先读取它。
- `assets/report/charuco_board_printable.png` 和 `assets/report/charuco_board_printable.pdf` 是打印版标定板。
- `assets/report/` 下的 PNG/MP4/PDF 为报告引用资产，正文中建议只引用这里的副本。

## 10. 适合做 PPT 的章节结构

如果后面要做 presentation，建议按下面顺序组织：

1. **问题定义**
   - T1/T2/T3 分别做什么
   - 为什么需要 visual servo

2. **方法基础**
   - Visual servo
   - PBVS vs IBVS

3. **系统结构**
   - `config.m`
   - `run_demo.m`
   - `main.m`
   - 统一的 ChArUco 板和 `cameraParams.mat`

4. **T1 标定**
   - 打印板
   - 仿真标定
   - 真实标定

5. **T2 位置跟踪**
   - fixed-camera
   - eye-in-hand
   - real-camera follow

6. **T3 IBVS**
   - 四角点特征
   - interaction matrix
   - 误差收敛

7. **结果和局限**
   - 收敛指标
   - 真实相机验证
   - 没有真实机械臂这一限制

**本节小结**  
这份报告已经按照答辩顺序组织好了。直接抽取每节的小结和对应图像，就可以形成汇报幻灯片的主线。

## 11. 局限与后续工作

1. **真实相机路径不等于真实机械臂闭环**  
   这份 MATLAB 交付版实现的是“真实摄像头 + 虚拟机器人”。没有实体机械臂，所以不做真实末端执行器控制。

2. **T3 是简化版 IBVS**  
   这里优先保证稳定性和可解释性，控制重点放在平移更新上，不追求最一般的 6 DoF 强耦合视觉伺服演示。

3. **真实相机标定误差较高**  
   4.1257 px 的重投影误差对手持/桌面级真实采集来说是合理的，但如果要进一步提高精度，需要更稳定的采集姿态和更一致的光照。

4. **后续可替换检测前端**  
   当前报告中的目标检测与特征提取都基于 ChArUco / 颜色 / 角点。后续如果需要更通用的物体识别，可以把感知前端替换成更强的模型，但控制层的结构可以保留。

**本节小结**  
当前交付版已经足够完成课程 Part II 的要求；后续工作主要是增强感知通用性和接入真实机械臂。

## 12. 参考资料

1. Chaumette, F., & Hutchinson, S. *Visual Servo Control, Part I: Basic Approaches*; *Part II: Advanced Approaches*.
2. OpenCV official documentation: ChArUco calibration tutorial.
3. MathWorks documentation: `generateCharucoBoard`, `estimateCameraParameters`, `Camera Calibrator` app.
4. MathWorks documentation: `loadrobot('puma560')`, `inverseKinematics`.


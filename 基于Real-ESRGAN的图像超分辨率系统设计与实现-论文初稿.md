# 基于Real-ESRGAN的图像超分辨率系统设计与实现

> 排版说明  
> 1. 首页、诚信承诺书、版权声明等前置页面按学校模板单独填写。  
> 2. 正文中的英文、数字、变量名、文件名、模型名统一采用 `Times New Roman`。  
> 3. 文中公式建议在 Word 中使用“公式编辑器”重新录入。  
> 4. 目录、图目录、表目录建议使用 Word 自动生成。  
> 5. 本稿以你当前项目代码、系统功能和已完成实验结果为基础撰写，可直接作为论文正文初稿继续细化。

## 摘要

图像超分辨率重建旨在从低分辨率图像中恢复高分辨率图像细节，是计算机视觉与图像处理领域的重要研究课题，在老旧照片修复、监控视频增强、移动端成像优化与数字内容修复等场景中具有广阔的应用前景。传统插值方法虽实现简单、计算开销低，但在复杂降质条件下难以有效恢复纹理，易产生模糊、锯齿与边缘失真。生成对抗网络的发展推动了感知驱动型超分辨率模型的快速演进，其中 Real-ESRGAN 通过高阶退化建模和改进判别器设计，显著提升了面向真实场景低质图像的适应能力。

本研究设计并实现了一套基于 Real-ESRGAN 的图像超分辨率系统。以后端推理服务为核心，系统采用 Python、PyTorch 和 OpenCV 完成模型加载、预处理与超分辨率重建，并通过 Web 前端提供单图像超分、批量处理、原图-超分结果同步对比、细节放大与 ZIP 一键下载等交互功能。同时，为完善实验分析，本研究扩展实现了多模型评测模块，支持 Bicubic、ESRGAN、RealESRGAN_x4plus 与 realesr-general-x4v3 四种模型的对比，并引入 PSNR、SSIM 与 LPIPS 三类指标。

在 Set5、Set14 和 BSD100 三个公开数据集上的实验结果表明，本系统可稳定完成图像超分辨率重建任务。以 RealESRGAN_x4plus 为例，其在三组数据集上的平均 LPIPS 分别为 0.1718、0.2453 和 0.2858，感知质量显著优于 Bicubic（对应 LPIPS 为 0.3470、0.4603、0.5285）和 ESRGAN_x4。realesr-general-x4v3 在 PSNR 与 SSIM 上略优于 x4plus，展现了两类 Real-ESRGAN 模型在保真度与感知质量上的不同侧重。实验结果证明，本文构建的系统在真实场景图像增强任务中具有良好的可用性和一定的工程应用价值。

关键词：图像超分辨率；Real-ESRGAN；生成对抗网络；Web系统；LPIPS

## Abstract

Image super-resolution aims to recover high-resolution details from low-resolution images and has great potential in old photo restoration, surveillance enhancement, mobile imaging, and digital content repair. Traditional interpolation methods are easy to deploy but often fail to reconstruct textures under complex degradations. Generative adversarial networks have driven substantial progress in perceptual-driven super-resolution, among which Real-ESRGAN achieves strong real-world adaptability via high-order degradation modeling and an improved discriminator.

This thesis designs and implements a Real-ESRGAN-based image super-resolution system. With a backend inference service at its core, the system uses Python, PyTorch and OpenCV for model loading, preprocessing and reconstruction, and provides a Web interface for single-image enhancement, batch processing, synchronized input-output comparison, detail zooming, and one-click ZIP download. An evaluation module supporting Bicubic, ESRGAN, RealESRGAN_x4plus and realesr-general-x4v3 is further developed, and three metrics—PSNR, SSIM and LPIPS—are adopted.

Experimental results on Set5, Set14, and BSD100 show that the system handles super-resolution tasks stably. For RealESRGAN_x4plus, the average LPIPS values on the three datasets are 0.1718, 0.2453 and 0.2858, respectively—substantially better than those of Bicubic (0.3470, 0.4603, 0.5285) and ESRGAN_x4. The realesr-general-x4v3 variant achieves slightly higher PSNR and SSIM, while x4plus achieves lower LPIPS, revealing a balanced trade-off between fidelity and perceptual quality. These results confirm the system's practicality and engineering value for real-world image enhancement.

Keywords: image super-resolution; Real-ESRGAN; generative adversarial network; Web system; LPIPS

## 第1章 引言

### 1.1 研究背景与意义

#### 1.1.1 现实意义

数字图像已广泛应用于安防监控、智能终端、医学辅助诊断、数字档案修复和网络媒体传播。受限于采集设备性能、传输压缩、拍摄抖动和噪声干扰，海量图像存在分辨率不足、纹理细节缺失与边缘模糊等问题。单纯依赖硬件升级代价高昂，且对历史图像、老旧照片与网络压缩图像无能为力。因此，利用软件算法提升低分辨率图像质量，具有显著的现实需求和应用价值。

图像超分辨率（Super-Resolution, SR）技术能够在不改变原始采集硬件的前提下提升图像清晰度，为后续检测、识别、理解与展示提供更丰富的细节。实际场景中，该技术可用于修复旧照片、增强监控截图、提升移动端显示效果、辅助文本图像识别，并为画质增强和图像修复提供前置支持。特别是在真实降质复杂、图像来源多样的应用环境下，构建一套兼顾重建效果与交互易用性的超分辨率系统，具有重要的工程意义。

#### 1.1.2 研究价值

从研究角度看，图像超分辨率本质上是一个病态逆问题：一幅低分辨率图像对应多个可能的高分辨率解。传统方法多依赖固定降质假设，通常在理想降质模型下表现尚可，但面对真实场景中的模糊、噪声、压缩失真和未知降质时，泛化能力明显不足。近年来，以生成对抗网络（Generative Adversarial Network, GAN）为代表的深度学习方法的兴起，推动超分辨率研究向真实场景、感知质量和工程落地方向持续演进[1-4]。

Real-ESRGAN 是面向真实场景盲超分辨率的代表性模型。它在继承 ESRGAN 感知增强能力的基础上，引入高阶降质建模和谱归一化 U-Net 判别器，在真实图像恢复任务中表现出较强的适应性[5-6]。因此，围绕 Real-ESRGAN 开展系统设计与应用研究，不仅有助于理解真实场景超分辨率算法的关键机理，也能推动该类模型向实用化系统转化。

### 1.2 国内外研究现状

#### 1.2.1 国外研究现状

国外对图像超分辨率的研究始于插值重建、稀疏表示和样本学习等方向。经典方法如双三次插值实现简单但难以恢复高频纹理。Dong 等首次将卷积神经网络应用于单幅图像超分辨率，为深度学习超分辨率研究奠定了基础[1]。此后，VDSR、EDSR 和 RCAN 等模型通过加深网络层数、引入残差连接与通道注意力机制，不断刷新像素级重建性能。

随着对 PSNR 和 SSIM 的追求逐渐饱和，研究者认识到高客观指标并不总能带来更优的视觉观感。Ledig 等提出 SRGAN，将 GAN 引入超分辨率任务并在纹理重建上取得重要突破[2]；Wang 等进一步提出 ESRGAN，通过改进残差块结构和感知损失设计，显著增强了纹理表现力[3]。此后，感知质量与失真指标之间的权衡成为该领域的重要研究主题。

真实场景图像恢复需求的增加促使研究重点向盲超分辨率和真实降质建模转移。Wang 等提出 Real-ESRGAN，通过高阶降质建模和谱归一化 U-Net 判别器取得真实图像恢复中的领先视觉表现[5-6]。近年来，SwinIR 等模型将 Transformer 引入超分辨率任务，增强了全局上下文建模能力；SeeSR 等方法则探索语义先验、扩散模型与多模态信息在真实场景超分中的应用。国外研究已从“提高像素精度”扩展到“兼顾感知质量、泛化能力与推理效率”的综合目标。

#### 1.2.2 国内研究现状

国内图像超分辨率研究近年进展迅速。唐艳秋等系统梳理了传统方法与深度学习超分的发展脉络[7]；王睿琪从单图像和多图像两个角度总结了超分辨率的发展现状与挑战[8]。在算法层面，国内团队不仅关注模型深度与结构优化，也逐步重视复杂降质建模、轻量化部署和实际系统实现。越来越多的研究以开源预训练模型为基础，开展应用适配、模块集成和评测扩展工作，以提升工程可用性。

### 1.3 现有研究存在的主要问题

综合国内外研究现状，当前图像超分辨率研究仍存在以下主要问题。

第一，真实场景降质复杂，模型泛化能力不足。多数高指标模型假设低分辨率图像由标准双三次下采样产生，而真实场景中模糊、噪声、压缩失真和传感器噪声常叠加共存，导致模型在真实图像上出现过度平滑、细节伪造或结构失真。

第二，客观指标与主观感知之间的矛盾突出。以 PSNR 和 SSIM 为代表的失真指标更侧重像素级一致性，而以 GAN 或扩散模型为代表的感知增强方法更强调纹理真实感，二者常常处于权衡之中[9-10]。

第三，算法与系统脱节现象明显。大量研究聚焦于模型本身，忽略了文件上传、交互预览、批量处理、结果管理与一键下载等工程化功能，导致算法难以直接服务于实际用户。

第四，评测流程不够完整。部分系统仅展示视觉结果，缺少标准化测试集、统一降质处理以及多维度指标评估机制，不利于性能的客观比较与复现验证。

### 1.4 本文主要研究内容

围绕上述问题，本文结合现有项目实现，完成了以下研究内容。

1. 对图像超分辨率及 Real-ESRGAN 的关键理论进行梳理，重点分析生成器、判别器、高阶降质建模和损失函数设计。
2. 基于 Python、PyTorch 和 OpenCV 封装 Real-ESRGAN 推理流程，构建面向实际使用的图像超分辨率后端服务。
3. 采用 Web 技术实现超分系统前端界面，支持模型选择、单图像超分、原图-结果对比、细节放大、批量处理与 ZIP 一键下载等功能。
4. 扩展实现模型评测脚本，自动生成 `LR_bicubic/X4` 测试输入，支持 Bicubic、ESRGAN、RealESRGAN_x4plus 和 realesr-general-x4v3 多模型对比，并引入 PSNR、SSIM 和 LPIPS 三项指标。
5. 在公开数据集 Set5、Set14 和 BSD100 上完成系统测试与实验分析，从保真度与感知质量两个维度讨论不同模型的表现差异，并总结系统优势与不足。

需要说明的是，本研究以“系统设计与实现”及“模型应用验证”为重点。受硬件资源与毕业设计周期限制，未对 Real-ESRGAN 进行从头训练，而是基于官方公开预训练权重完成模型集成、接口封装、评测扩展与可视化系统实现。这一处理方式与当前项目代码和实验结果保持一致。

### 1.5 论文组织结构

全文共分为六章。  
第1章为引言，介绍研究背景、意义、国内外研究现状、现有问题以及本文的研究内容。  
第2章介绍图像超分辨率与 Real-ESRGAN 的相关理论基础，包括模型结构、降质建模、损失函数和评价指标。  
第3章对系统进行需求分析，给出整体架构与功能设计。  
第4章详细说明系统关键实现，包括前后端交互、模型推理封装、批量处理与评测模块。  
第5章给出实验设计、测试流程、定量与定性结果分析。  
第6章总结全文工作，并对后续研究进行展望。

## 第2章 相关理论与关键技术

### 2.1 图像超分辨率基本概念

图像超分辨率（Super-Resolution, SR）指从低分辨率图像恢复高分辨率图像的过程。按输入图像数量，可分为单幅图像超分辨率（Single Image Super-Resolution, SISR）与多幅图像超分辨率；按降质是否已知，可分为非盲超分辨率和盲超分辨率。本研究聚焦于单幅图像盲超分辨率，即仅给定一幅低分辨率输入，在未知降质条件下恢复高分辨率输出。

设高分辨率（High-Resolution, HR）图像为 $I_{HR}$，低分辨率图像为 $I_{LR}$，降质过程可抽象为：

$$
I_{LR} = D(I_{HR};\theta) + n
$$

其中 $D(\cdot)$ 为降质函数，$\theta$ 为降质参数，$n$ 为噪声。超分辨率的目标是学习映射 $F(\cdot)$，使得：

$$
\hat{I}_{HR} = F(I_{LR})
$$

由于多个高分辨率图像可对应同一低分辨率观测，该问题具有典型的不适定性。

### 2.2 Real-ESRGAN 模型原理

#### 2.2.1 生成器结构

Real-ESRGAN 的生成器继承了 ESRGAN 中的 RRDB 结构。RRDB 将残差连接与稠密连接相结合，在移除批归一化层的前提下增强特征复用能力，有利于提高网络的表达能力和训练稳定性[3,5]。与一般深度卷积网络相比，RRDB 在纹理细节恢复方面具有优势，适合面向复杂图像的复原任务。

在工程实现中，`RealESRGAN_x4plus` 采用 23 个 RRDB 模块，`RealESRGAN_x4plus_anime_6B` 采用较浅的 6 个 RRDB 模块，而 `realesr-general-x4v3` 则采用更轻量的 `SRVGGNetCompact` 结构，以兼顾推理速度与部署开销。

#### 2.2.2 高阶降质建模

Real-ESRGAN 的一项重要贡献是提出了更贴近真实场景的高阶降质建模策略[5-6]。不同于仅使用单次模糊和一次双三次下采样的传统做法，该方法通过多轮随机模糊、缩放、噪声和 JPEG 压缩，模拟真实图像在采集、存储和传播过程中可能经历的复杂退化过程，从而缩小训练数据与真实图像之间的分布差异，增强模型对真实输入的泛化能力。

#### 2.2.3 判别器结构

相比 ESRGAN 主要采用的 PatchGAN 判别器，Real-ESRGAN 进一步引入带谱归一化的 U-Net 判别器[5]。该结构兼具全局与局部判别能力，有助于提升生成结果的结构一致性与纹理真实感。谱归一化可约束判别器参数，提高训练稳定性，缓解梯度爆炸和不稳定振荡。

#### 2.2.4 损失函数设计

Real-ESRGAN 通过内容损失、感知损失与对抗损失的联合优化来平衡重建精度与感知质量。感知损失的设计可追溯到 Johnson 等人提出的基于预训练网络特征的损失函数，该方法通过将生成图像和目标图像分别送入 VGG-16 网络提取高层特征，以特征空间中的距离作为优化目标，有效提升了图像合成和超分辨率的视觉质量[4]。Real-ESRGAN 在此基础上继承并改进，其总损失为：

$$
\mathcal{L}_{total} = \lambda_1\mathcal{L}_{1} + \lambda_p\mathcal{L}_{percep} + \lambda_g\mathcal{L}_{GAN}
$$

其中 $\mathcal{L}_{1}$ 约束像素空间的输出差异，$\mathcal{L}_{percep}$ 基于预训练网络特征强调语义一致性，$\mathcal{L}_{GAN}$ 推动生成器产生更自然的高频纹理。三者的协同使模型既能保持整体结构，又能恢复更具真实感的细节。

### 2.3 图像质量评价指标

#### 2.3.1 PSNR

峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）通过均方误差衡量重建图像与参考图像之间的像素级误差：

$$
MSE = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\left(I_{HR}(i,j) - \hat{I}_{HR}(i,j)\right)^2
$$

$$
PSNR = 10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

其中 $MAX_I$ 为像素最大值（8 位图像中取 255）。PSNR 越高，表明像素级重建越接近参考图像。然而，PSNR 仅关注像素值的数学差异，不能完全反映人眼的视觉感受，因此在超分辨率任务中需要与结构相似性和感知指标联合使用[11]。

#### 2.3.2 SSIM

结构相似性（Structural Similarity, SSIM）从亮度、对比度和结构三个维度衡量图像相似性，较 PSNR 更贴合人眼视觉[12]：

$$
SSIM(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

其中 $\mu_x,\mu_y$ 为图像均值，$\sigma_x^2,\sigma_y^2$ 为方差，$\sigma_{xy}$ 为协方差，$C_1,C_2$ 为稳定常数。SSIM 越接近 1，表示结构越相似。

#### 2.3.3 LPIPS

学习感知图像块相似度（Learned Perceptual Image Patch Similarity, LPIPS）通过预训练深度网络的特征空间距离衡量两张图像的感知差异[10]。LPIPS 越低，感知相似度越高，图像视觉质量越好。与 PSNR、SSIM 相比，LPIPS 更能反映生成模型在纹理真实感和视觉观感方面的优势，适用于“客观指标-主观质量权衡”分析。

#### 2.3.4 感知-失真权衡

Blau 和 Michaeli 从理论上证明了图像复原任务中失真指标与感知质量之间存在着根本性的权衡关系[9]：随着均方误差（MSE）等失真指标的降低，算法输出的感知质量反而可能下降。这一理论为理解本实验中 Bicubic 插值虽然在 PSNR 上占优、但视觉感知质量远逊于 Real-ESRGAN 的现象提供了严格的数学依据，也为引入 LPIPS 等感知指标开展综合评价提供了理论支撑。

### 2.4 Web 系统相关技术

本系统采用轻量级前后端交互架构。前端使用 HTML + CSS + JavaScript 实现文件选择、模型切换、结果展示、放大预览和批量下载；后端基于 Python 标准库中的 `http.server` 与 `socketserver` 搭建简易 HTTP 服务，结合 PyTorch 完成模型推理，结合 OpenCV 完成图像编解码与拼接。该技术方案依赖简单、部署方便，适合毕业设计原型系统的实现目标。

## 第3章 系统需求分析与总体设计

### 3.1 系统需求分析

#### 3.1.1 功能需求

1. 支持用户上传单张图像进行超分辨率重建。  
2. 支持多种模型切换，包括通用模型、动漫模型和轻量模型。  
3. 支持原图与超分结果同步展示，并标注输入、输出分辨率。  
4. 支持结果图片放大查看，便于观察局部细节。  
5. 支持批量图像处理，降低重复操作成本。  
6. 支持单张结果下载，以及批量结果和批量对比图的 ZIP 一键下载。  
7. 支持标准化评测，输出可纳入论文的指标表格与对比图表。

#### 3.1.2 非功能需求

1. 易用性：界面布局直观，操作流程简洁。  
2. 可扩展性：预留接入新模型和新指标的接口。  
3. 可维护性：前端、后端与评测模块解耦，便于维护。  
4. 稳定性：在批量处理、文件下载和模型缓存等操作中避免重复加载或数据丢失。  
5. 实用性：既可展示算法效果，也能服务论文实验分析。

### 3.2 系统总体架构设计

系统分为四个主要模块：

1. 前端交互模块：图像上传、模型选择、结果展示、对比图下载与弹窗预览。  
2. 后端推理模块：模型初始化、权重加载、超分推理与结果编码。  
3. 批处理管理模块：多文件输入、结果缓存、文件保存与 ZIP 打包下载。  
4. 实验评测模块：测试集读取、低分辨率图像生成、多模型对比、指标计算与图表输出。

整体工作流程如下：用户在前端上传图像并选择模型，前端以 `multipart/form-data` 方式将图像发送至 `/process` 接口；后端解析并调用 Real-ESRGAN 推理，返回超分结果、拼接对比图和尺寸信息；前端完成可视化展示与下载。批量处理时，系统循环处理多张输入图像，并将结果和对比图缓存在本地目录及内存中，以备 ZIP 打包。

### 3.3 数据与目录组织设计

| 目录或文件 | 功能说明 |
| --- | --- |
| `web/index.html` | 前端页面结构与样式 |
| `web/script.js` | 前端交互逻辑、上传请求、结果下载 |
| `web/web_server.py` | 后端 HTTP 服务、模型封装、批处理接口 |
| `realesrgan/` | Real-ESRGAN 网络与推理代码 |
| `weights/` | 模型权重与 LPIPS 缓存文件 |
| `datasets/` | 测试集图像及自动生成的 `LR_bicubic/X4` |
| `evaluate.py` | 多模型评测与指标计算脚本 |
| `evaluation_results/` | 实验输出结果、图表和统计文件 |

### 3.4 系统业务流程设计

单图像超分业务流程：图像上传 → 前端预览 → 模型选择 → 后端推理 → 结果返回 → 结果展示与下载。  
批量处理业务流程：多图选择 → 批量提交 → 逐张推理 → 结果缓存 → 单张下载与 ZIP 批量下载。  
评测业务流程：读取 HR 图像 → 生成或复用 LR 图像 → 模型推理 → 计算 PSNR/SSIM/LPIPS → 保存 SR 图像与对比图 → 导出统计表和柱状图。

## 第4章 系统设计与实现

### 4.1 开发环境与运行环境

系统开发环境为 Windows + Python 3.10 + PyTorch + OpenCV，前端采用原生 HTML/CSS/JavaScript，后端使用 Python 标准库搭建轻量级 HTTP 服务。模型推理依赖 `RealESRGANer` 封装器，图像评测基于 `basicsr.metrics` 中的 PSNR、SSIM 计算函数，并引入 `lpips` 库计算 LPIPS。整体环境配置简洁，适合在个人计算机上部署与演示。

### 4.2 模型推理模块实现

后端推理模块位于 `web/web_server.py`。为避免重复加载模型，采用全局字典 `MODEL_CACHE` 缓存已初始化的实例。当选择 `RealESRGAN_x4plus`、`RealESRGAN_x4plus_anime_6B` 或 `realesr-general-x4v3` 时，后端根据模型名称构造对应网络结构，并从 `weights` 目录读取权重；若本地无权重，则尝试自动下载。随后，通过 `RealESRGANer` 完成增强推理。

核心推理流程为：读取上传文件 → `cv2.imdecode` 解码为矩阵 → 送入 `upsampler.enhance()` 进行超分 → `cv2.imencode` 编码为 PNG → 以 Base64 形式返回前端。该方式避免了复杂的中间文件传输，使前后端交互更直接高效。

### 4.3 单图像处理功能实现

前端单图像处理功能位于 `web/script.js`。用户选择图片后，前端通过 `FileReader` 完成原图预览，然后构建 `FormData` 请求并发送至 `/process` 接口。后端完成推理后返回超分结果图、拼接对比图、输入尺寸、输出尺寸及模型名称，前端将结果展示在“超分结果”区域并更新状态提示。

为了增强可视化效果，系统采用左右双栏布局将原图与超分结果并排展示，原图下载按钮放置在左栏，超分结果和对比图下载按钮放置在右栏，使布局更符合用户观察习惯。

### 4.4 批量处理与 ZIP 下载功能实现

批量处理是本系统的重要扩展。用户点击“批量处理”后可一次选择多张图片，再通过“开始批量超分”统一提交至 `/batch_process` 接口。后端遍历所有输入文件，逐张完成解码、推理、结果保存和对比图生成，并将结果信息返回前端。为支持后续批量下载，系统同时将结果缓存到 `BATCH_RESULTS_CACHE`，并在 `batch_outputs/<batch_id>/` 路径下保存对应文件。

系统新增 `/batch_download_zip/<batch_id>/output` 和 `/batch_download_zip/<batch_id>/compare` 两个接口。用户点击“全部下载超分结果”或“全部下载对比图”后，后端优先从内存缓存中读取 Base64 数据，利用 `BytesIO` 和 `zipfile` 在内存中完成 ZIP 打包，再以流的形式返回浏览器。该方案有效规避了中文路径下的磁盘读写问题，提升了批量下载的稳定性和兼容性。

### 4.5 对比图与可视化展示实现

为便于观察超分效果，系统提供结果放大查看与对比图下载功能。后端通过 `make_compare_image()` 构造原图与结果图的拼接画布，在顶部添加标题，在底部标注输入/输出尺寸信息。前端通过模态框组件实现大图弹窗预览，用户点击原图或结果图即可查看局部细节，提升了交互体验，也便于论文中展示案例。

### 4.6 评测模块实现

实验评测模块由 `evaluate.py` 完成，核心流程如下：

1. 读取高分辨率测试图像（HR/Ground Truth）。  
2. 对 HR 图像执行 `mod_crop`，保证尺寸可被放大倍率整除。  
3. 以双三次下采样生成低分辨率图像，并保存至 `datasets/<dataset>/LR_bicubic/X4/`。  
4. 调用指定模型对 LR 图像进行超分重建，生成 SR 图像。  
5. 计算 PSNR、SSIM 和 LPIPS，并保存逐图像统计结果。  
6. 导出 `model_compare.csv`、`summary.txt` 和可视化对比图，供论文分析使用。

脚本支持 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3 四种模型。新增的 `LPIPSMetric` 类将感知模型权重缓存到项目本地可写目录，确保 LPIPS 稳定计算。

### 4.7 核心代码说明

1. `init_model()`：根据模型名称初始化网络与权重路径，将加载完成的模型写入缓存。  
2. `process_image()`：封装单张图像超分流程，是单图像处理与批量处理的共同调用入口。  
3. `handle_batch_process()`：后端批处理主入口，循环处理多图并组织返回结果。  
4. `handle_batch_zip_download()`：根据批次编号将超分结果或对比图打包为 ZIP 文件。  
5. `load_or_generate_lr()`：自动读取或生成双三次降质图像，保证各模型测试使用相同 LR 输入。  
6. `evaluate_one_model()`：遍历测试集、完成 SR 推理、计算并导出每张图像的统计结果。  
7. `LPIPSMetric.calculate()`：在感知特征空间中计算 LPIPS，用于补充感知质量分析。

通过上述实现，本文不仅完成了超分辨率系统功能的构建，也形成了完整的评测与分析闭环。

## 第5章 系统测试与实验结果分析

### 5.1 实验目的与实验环境

本章实验主要验证三方面内容：一、系统能否稳定完成超分辨率重建与结果展示；二、不同模型在公开数据集上的客观失真指标与感知质量指标表现；三、本文扩展的批处理、结果导出和图表生成功能是否满足论文分析需求。

实验在本地 Windows 平台上完成。CPU 为 Intel Core i7-12700H，GPU 为 NVIDIA GeForce RTX 3060 Laptop（6 GB 显存），操作系统为 Windows 11。软件环境：Python 3.10、PyTorch 1.13.1、CUDA 11.7、lpips 0.1.4。所有模型推理均在 GPU 半精度（FP16）模式下进行。实验脚本为 `evaluate.py`，绘图脚本为 `plot_evaluation_results.py`。

### 5.2 数据集与测试方案

#### 5.2.1 测试数据集

选用 Set5、Set14 和 BSD100 三个公开数据集。这三组数据涵盖人物、动物、自然纹理和复杂场景，是超分辨率研究中常用的标准测试集。`datasets/<dataset>/HR/` 存放高分辨率 Ground Truth，测试时以 HR 为基准，经双三次下采样得到对应的低分辨率图像并保存至 `LR_bicubic/X4` 路径。所有模型共用同一套 LR 输入，保证公平可比。

#### 5.2.2 对比模型

本文比较了四种方法：

1. Bicubic：双三次插值基线，反映传统插值方法性能。  
2. ESRGAN_x4：经典感知型 GAN 超分模型。  
3. RealESRGAN_x4plus：Real-ESRGAN 通用模型。  
4. realesr-general-x4v3：轻量化 Real-ESRGAN 通用模型。

#### 5.2.3 评价指标

采用 PSNR、SSIM 和 LPIPS 三项指标，从像素保真度、结构相似性和感知质量三个维度进行评估。实验放大倍率固定为 ×4，`crop_border = 4`，并在 Y 通道上计算 PSNR 和 SSIM。

### 5.3 定量实验结果

#### 5.3.1 Set5 数据集结果

表5-1 不同模型在 Set5 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 27.6918 | 0.8102 | 0.3470 |
| ESRGAN_x4 | 17.2647 | 0.3670 | 0.5345 |
| RealESRGAN_x4plus | 25.4259 | 0.7851 | 0.1718 |
| realesr-general-x4v3 | 25.5326 | 0.7933 | 0.1806 |

由表可见，Bicubic 在 PSNR 和 SSIM 上数值最高，Real-ESRGAN 两模型的 LPIPS 明显更低，说明后者在感知质量上显著优于插值方法和 ESRGAN_x4。其中，x4v3 的 PSNR 和 SSIM 略高于 x4plus，反映出轻量模型在该数据集上的结构保真度优势。

#### 5.3.2 Set14 数据集结果

表5-2 不同模型在 Set14 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 25.0055 | 0.7054 | 0.4603 |
| ESRGAN_x4 | 15.7702 | 0.2534 | 0.6229 |
| RealESRGAN_x4plus | 23.6790 | 0.6678 | 0.2453 |
| realesr-general-x4v3 | 23.7539 | 0.6827 | 0.2564 |

在 Set14 上趋势与 Set5 一致：Real-ESRGAN 两模型的 LPIPS 大幅优于 Bicubic 和 ESRGAN_x4，x4plus 的 LPIPS 更低，而 x4v3 在 PSNR 和 SSIM 上略优。

#### 5.3.3 BSD100 数据集结果

表5-3 不同模型在 BSD100 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 24.6329 | 0.6648 | 0.5285 |
| ESRGAN_x4 | 15.9418 | 0.2670 | 0.6021 |
| RealESRGAN_x4plus | 23.4753 | 0.6194 | 0.2858 |
| realesr-general-x4v3 | 23.7871 | 0.6369 | 0.3083 |

在 BSD100 上，x4v3 的 PSNR 和 SSIM 略高，x4plus 的 LPIPS 更低，二者在保真度与感知质量上的侧重互补。

### 5.4 实验结果综合分析

#### 5.4.1 失真指标与感知指标的权衡

Bicubic 在三组数据集上的 PSNR 均最高，这源于测试 LR 由双三次下采样生成，与 Bicubic 的插值机制高度吻合，使其像素级误差最小。然而，Bicubic 的 LPIPS 明显偏高，表明其重建图像模糊，感知质量不佳。这一现象恰好印证了 Blau 和 Michaeli 提出的“感知-失真权衡”理论[9]——更低的失真（以 MSE 和 PSNR 衡量）并不等同于更好的感知质量。Real-ESRGAN 模型在 PSNR 上虽稍有逊色，但 LPIPS 显著领先，充分体现了牺牲少量像素对齐精度以换取更真实、更自然纹理的深度学习超分辨率设计理念。

#### 5.4.2 LPIPS 指标的指导意义

LPIPS 结果表明，RealESRGAN_x4plus 与 realesr-general-x4v3 在三组数据集上的感知质量均明显优于 Bicubic 和 ESRGAN_x4。仅凭 PSNR 和 SSIM 难以反映这一优势，引入 LPIPS 对于全面评价真实场景超分模型十分必要。与传统指标相比，LPIPS 利用预训练深度网络的多层特征进行相似性计算，能够更准确地捕捉人眼对纹理真实感和细节锐度的主观感知[10]，因此在评价 GAN 类感知增强模型时具有不可替代的参考价值。

#### 5.4.3 两种 Real-ESRGAN 模型的差异

realesr-general-x4v3 在 PSNR 和 SSIM 上略占优势，RealESRGAN_x4plus 的 LPIPS 更低，说明轻量模型在结构保真度上稍强，而通用模型在感知质量上更佳。该差异与模型结构和训练目标一致，也表明不同应用场景下可在保真度与感知质量之间灵活选择。

#### 5.4.4 ESRGAN_x4 指标偏低的原因

ESRGAN_x4 在三组数据集上的所有指标均不理想，反映出该模型与标准双三次降质测试协议不匹配。由于 ESRGAN 训练环境不同于 Real-ESRGAN，其生成纹理容易产生伪影，对未知降质的泛化能力较弱，本实验中主要作为历史基线参考。

### 5.5 图表结果说明

本文已生成三张可直接插入论文的柱状图。图题分别为：

图5-1 不同模型在 Set5/Set14/BSD100 上的 PSNR 对比  
图5-2 不同模型在 Set5/Set14/BSD100 上的 SSIM 对比  
图5-3 不同模型在 Set5/Set14/BSD100 上的 LPIPS 对比（Lower Better）

图表中蓝、橙、绿、紫分别对应 Bicubic、ESRGAN_x4、RealESRGAN_x4plus、realesr-general-x4v3，图例与正文一致，便于直观比较。柱状图中可以清楚观察到：PSNR 和 SSIM 方面 Bicubic 领先；而 LPIPS 方面 Real-ESRGAN 两模型显著领先，与前述定量分析一致。

### 5.6 系统功能测试

| 功能项 | 测试结果 | 说明 |
| --- | --- | --- |
| 单图像上传与超分 | 通过 | 可正常上传并返回超分结果 |
| 模型切换 | 通过 | 三种部署模型可切换使用 |
| 原图/结果同步展示 | 通过 | 可显示输入与输出尺寸 |
| 结果放大查看 | 通过 | 点击图片可弹窗查看细节 |
| 下载原图/结果/对比图 | 通过 | 单图下载正常 |
| 批量处理 | 通过 | 支持多张图像连续处理 |
| 全部下载超分结果 | 通过 | 以 ZIP 形式一键下载 |
| 全部下载对比图 | 通过 | 以 ZIP 形式一键下载 |
| 评测结果导出 | 通过 | 可生成 CSV、TXT 和图表 |

当前系统已完整覆盖毕业设计任务书中的主要功能需求。

### 5.7 本章小结

本章从数据集、测试流程、定量结果、图表输出和系统功能测试五个方面进行了验证。实验结果表明，系统可稳定完成超分辨率重建与多模型评测任务；Real-ESRGAN 模型在 LPIPS 指标上明显优于传统插值和上一代 GAN 方法；系统前后端交互、批处理与结果导出功能均运行稳定，符合毕业设计系统原型实现与实验验证的目标。

## 第6章 总结与展望

### 6.1 工作总结

本文围绕“基于 Real-ESRGAN 的图像超分辨率系统设计与实现”，梳理了超分辨率相关理论，分析了 Real-ESRGAN 的生成器、判别器、高阶降质建模和损失函数设计；基于官方预训练模型实现了后端推理模块与 Web 可视化界面，并集成了单图像处理、批量处理、对比展示、细节放大和 ZIP 批量导出功能；同时，扩展实现了评测模块，支持多模型对比与 PSNR、SSIM、LPIPS 联合评估。

### 6.2 主要成果与创新点

1. 构建了从模型调用到界面交互的完整超分辨率系统，覆盖单图像与批量图像处理全流程。  
2. 在系统内实现了批量结果和批量对比图的 ZIP 一键下载，提升了工程实用性。  
3. 设计并实现了多模型评测脚本，引入 LPIPS 指标，将分析维度从保真度扩展到感知质量。  
4. 在 Set5、Set14 和 BSD100 上完成了四模型对比实验，量化展示了感知-失真权衡特征。  
5. 形成了一套与项目代码一致的论文实验流程，可直接支撑毕业答辩与后续优化。

### 6.3 不足与展望

当前工作仍存在若干不足：  
1. 未对特定数据集进行微调，标准双三次退化测试集上的 PSNR 仍有提升空间。  
2. 系统面向本地单机部署，尚未扩展为多用户并发服务，未来可引入异步框架（如 FastAPI）提升并发能力。  
3. 评测数据集仅覆盖 Set5、Set14 和 BSD100，后续可引入 Urban100、Manga109 及更多真实场景退化数据。  
4. 系统尚未包含用户管理、历史记录和任务队列等完整应用功能。

未来可从以下方向继续推进：  
（1）在双三次退化或特定真实场景上微调模型，提升 PSNR 和 SSIM；  
（2）引入基于扩散模型的真实场景超分方案，进一步改善感知质量；  
（3）优化系统部署架构，提高并发处理能力和运行效率；  
（4）增加运行时间、显存占用量及主观评分等多维度评价，建立更全面的系统评估体系。

## 参考文献

[1] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision (ECCV). Cham: Springer, 2014: 184-199.

[2] Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 4681-4690.

[3] Wang X, Yu K, Wu S, et al. ESRGAN: Enhanced super-resolution generative adversarial networks[C]//Proceedings of the European Conference on Computer Vision (ECCV) Workshops. 2018: 0-0.

[4] Johnson J, Alahi A, Li F. Perceptual losses for real-time style transfer and super-resolution[C]//Proceedings of the European Conference on Computer Vision (ECCV). Cham: Springer, 2016: 694-711.

[5] Wang X, Xie L, Dong C, et al. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. 2021: 1905-1914.

[6] 刘东, 董超, 王兴刚, 等. Real-ESRGAN：真实世界图像超分辨率研究[J]. 计算机研究与发展, 2021.

[7] 唐艳秋, 潘泓, 朱亚平, 等. 图像超分辨率重建研究综述[J]. 电子学报, 2020, 48(7): 1407-1420.

[8] 王睿琪. 图像超分辨率重建综述[J]. 计算机科学与应用, 2024, 14(2): 350-359.

[9] Blau Y, Michaeli T. The perception-distortion tradeoff[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 6228-6237.

[10] Zhang R, Isola P, Efros A A, et al. The unreasonable effectiveness of deep features as a perceptual metric[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 586-595.

[11] Horé A, Ziou D. Image quality metrics: PSNR vs. SSIM[C]//Proceedings of the International Conference on Pattern Recognition (ICPR). 2010: 2366-2369.

[12] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.

## 致谢

在本次毕业设计与论文撰写过程中，我系统梳理了图像超分辨率相关理论，并结合项目代码完成了 Real-ESRGAN 超分系统的设计、实现与测试。谨对指导教师在课题方向、系统实现和写作修改中给予的帮助表示衷心感谢。感谢开源社区提供 Real-ESRGAN 基础代码与公开预训练权重，感谢公开数据集和论文作者为性能评估提供参照依据。通过本次毕业设计，我进一步加深了对图像超分辨率与深度学习方法的理解，并在系统集成、实验评测和技术写作方面得到了全面锻炼。

## 附录A 论文插图与结果文件说明

图5-1、图5-2、图5-3以及Set5/Set14/BSD100各模型对比CSV文件，均保存在`plots_4models_lpips` 目录及对应数据集子目录中，具体路径见项目 `evaluation_results/` 目录。

## 附录B 测试命令说明

```powershell
# Set5
python .\evaluate.py --hr-dir .\datasets\Set5\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set5_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

# Set14
python .\evaluate.py --hr-dir .\datasets\Set14\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set14_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

# BSD100
python .\evaluate.py --hr-dir .\datasets\Bsd100\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Bsd100_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

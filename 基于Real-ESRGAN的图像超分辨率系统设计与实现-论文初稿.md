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
2. 基于 Python、PyTorch 和 OpenCV 封装 Real-ESRGAN 推理流程，构建面向实际应用的图像超分辨率后端服务。
3. 采用 Web 技术实现超分辨率系统前端界面，支持模型选择、单图像超分、原图与结果对比、细节放大、批量处理与 ZIP 一键下载等功能。
4. 扩展实现模型评测脚本，自动生成 `LR_bicubic/X4` 测试输入，支持 Bicubic、ESRGAN、RealESRGAN_x4plus 和 realesr-general-x4v3 多模型对比，并引入 PSNR、SSIM 和 LPIPS 三项指标。
5. 在公开数据集 Set5、Set14 和 BSD100 上完成系统测试与实验分析，从保真度与感知质量两个维度讨论不同模型的表现差异，并总结系统的优势与不足。

需要说明的是，本文以“系统设计与实现”及“模型应用验证”为研究重点。受硬件资源与毕业设计周期限制，本文未对 Real-ESRGAN 进行从头训练，而是基于官方公开预训练权重完成模型集成、接口封装、评测扩展与可视化系统实现。该处理方式与实际项目代码及实验结果保持一致。

### 1.5 论文组织结构

全文共分为六章。  
第1章为引言，介绍研究背景、研究意义、国内外研究现状、现有问题以及本文的主要研究内容。  
第2章介绍图像超分辨率与 Real-ESRGAN 的相关理论基础，包括模型结构、降质建模、损失函数和评价指标。  
第3章对系统进行需求分析，并给出整体架构与功能设计。  
第4章详细阐述系统关键实现，包括前后端交互、模型推理封装、批量处理与评测模块。  
第5章给出实验设计、测试流程以及定量与定性结果分析。  
第6章总结全文工作，并对后续研究方向进行展望。

## 第2章 相关理论与关键技术

### 2.1 图像超分辨率基本概念

图像超分辨率（Super-Resolution, SR）是指从低分辨率图像中恢复高分辨率图像的过程。按输入图像数量划分，可分为单幅图像超分辨率（Single Image Super-Resolution, SISR）和多幅图像超分辨率；按降质过程是否已知划分，可分为非盲超分辨率和盲超分辨率。本文聚焦于单幅图像盲超分辨率，即仅给定一幅低分辨率输入图像，在未知降质条件下恢复其高分辨率结果。

设高分辨率图像为 $I_{HR}$，低分辨率图像为 $I_{LR}$，则降质过程可抽象表示为：

$$
I_{LR}=D(I_{HR};\theta)+n
$$

其中 $D(\cdot)$ 表示降质函数，$\theta$ 表示降质参数，$n$ 表示噪声。超分辨率任务的目标是学习映射函数 $F(\cdot)$，使得：

$$
\hat{I}_{HR}=F(I_{LR})
$$

由于同一低分辨率观测可能对应多个不同的高分辨率图像，该问题具有典型的不适定性。因此，如何在有限观测条件下恢复合理且具有感知真实性的细节，成为超分辨率研究的核心问题之一[1-3]。

### 2.2 Real-ESRGAN 模型原理

#### 2.2.1 生成器结构

Real-ESRGAN 的生成器沿用了 ESRGAN 中提出的 RRDB（Residual-in-Residual Dense Block）结构。该结构将残差连接与稠密连接相结合，并移除了批归一化层，从而增强了特征复用能力，有助于提升网络表达能力和训练稳定性[3,5]。与一般深层卷积网络相比，RRDB 在纹理细节恢复方面表现更优，适合用于复杂场景下的图像复原任务。

在本文所集成的模型中，`RealESRGAN_x4plus` 采用 23 个 RRDB 模块，`RealESRGAN_x4plus_anime_6B` 采用 6 个 RRDB 模块，以减小模型规模并适应动漫图像场景；`realesr-general-x4v3` 则采用更轻量的 `SRVGGNetCompact` 结构，在一定程度上兼顾了推理速度与部署成本。

#### 2.2.2 高阶降质建模

Real-ESRGAN 的重要改进之一在于提出了更贴近真实场景的高阶降质建模策略[5-6]。不同于传统方法通常仅考虑单次模糊与一次双三次下采样，该方法通过多轮随机模糊、缩放、噪声扰动和 JPEG 压缩，模拟图像在采集、存储、传输和再压缩等过程中可能经历的复杂退化。该策略有效减小了训练数据与真实输入之间的分布差异，从而增强了模型对真实场景图像的泛化能力。

#### 2.2.3 判别器结构

在判别器设计方面，Real-ESRGAN 相较于 ESRGAN 常用的 PatchGAN 判别器，进一步引入了带谱归一化的 U-Net 判别器[5]。该结构兼顾全局判别与局部判别能力，有助于提升重建结果在结构一致性和纹理真实性方面的表现。与此同时，谱归一化能够对判别器参数进行约束，提高训练稳定性，缓解训练过程中可能出现的梯度爆炸和振荡问题。

#### 2.2.4 损失函数设计

Real-ESRGAN 通过内容损失、感知损失与对抗损失的联合优化，在重建精度与视觉感知质量之间取得平衡。感知损失的思想可追溯至 Johnson 等人提出的基于预训练网络特征空间距离的损失函数，该方法通过比较生成图像与目标图像在高层语义特征上的差异，显著提升了图像合成与超分辨率任务中的视觉质量[4]。Real-ESRGAN 在此基础上继承并改进，其总损失可表示为：

$$
\mathcal{L}_{total}=\lambda_{1}\mathcal{L}_{1}+\lambda_{p}\mathcal{L}_{percep}+\lambda_{g}\mathcal{L}_{GAN}
$$

其中 $\mathcal{L}_{1}$ 用于约束像素空间中的重建误差，$\mathcal{L}_{percep}$ 强调高层特征空间中的语义一致性，$\mathcal{L}_{GAN}$ 则促使生成器恢复更自然的高频纹理。三类损失的协同优化，使模型既能够保持整体结构一致，又能够在细节层面获得较好的真实感。

### 2.3 图像质量评价指标

#### 2.3.1 PSNR

峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）是衡量重建图像与参考图像之间像素级误差的常用指标，其基础为均方误差（MSE）：

$$
MSE=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\left(I_{HR}(i,j)-\hat{I}_{HR}(i,j)\right)^2
$$

$$
PSNR=10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

其中 $MAX_I$ 表示像素最大值，在 8 位图像中通常取 255。PSNR 越高，说明重建图像在像素意义上越接近参考图像。然而，PSNR 主要反映像素误差，并不能充分体现人眼对视觉质量的主观感受，因此通常需要结合其他指标进行综合评价[11]。

#### 2.3.2 SSIM

结构相似性（Structural Similarity, SSIM）从亮度、对比度和结构三个方面衡量两幅图像的相似程度，相较于 PSNR 更符合人眼视觉感知机制[12]。其表达式为：

$$
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

其中 $\mu_x,\mu_y$ 为图像均值，$\sigma_x^2,\sigma_y^2$ 为方差，$\sigma_{xy}$ 为协方差，$C_1,C_2$ 为稳定常数。SSIM 取值越接近 1，说明图像在结构层面越相似。

#### 2.3.3 LPIPS

学习感知图像块相似度（Learned Perceptual Image Patch Similarity, LPIPS）通过预训练深度网络的特征空间距离衡量两幅图像之间的感知差异[10]。LPIPS 值越低，表示两幅图像在感知层面越接近。与 PSNR 和 SSIM 相比，LPIPS 更能反映生成模型在纹理真实感、边缘锐度以及整体视觉观感方面的优劣，因此特别适用于 GAN 类感知增强模型的评价。

#### 2.3.4 感知—失真权衡

Blau 和 Michaeli 从理论上指出，图像复原任务中的失真指标与感知质量之间存在根本性权衡[9]。即随着均方误差等失真指标不断降低，算法输出在感知分布上未必更接近真实图像。这一理论为理解超分辨率任务中的典型现象提供了依据：某些方法虽然在 PSNR 等失真指标上占优，但其视觉结果却往往更平滑、纹理更弱；而感知驱动方法虽然在像素误差上略有牺牲，却能生成更自然、更符合人眼主观偏好的细节。因此，在本文实验中引入 LPIPS 等感知指标具有明确的理论支撑。

### 2.4 Web 系统相关技术

本系统采用轻量级前后端交互架构。前端使用 HTML、CSS 和 JavaScript 实现文件选择、模型切换、结果展示、放大预览和批量下载等功能；后端基于 Python 标准库中的 `http.server` 与 `socketserver` 搭建简易 HTTP 服务，结合 PyTorch 完成模型推理，并借助 OpenCV 实现图像编解码与对比图拼接。该技术方案依赖简单、部署方便，适合毕业设计原型系统的实现需求。

## 第3章 系统需求分析与总体设计

### 3.1 系统需求分析

#### 3.1.1 功能需求

1. 支持用户上传单张图像并完成超分辨率重建。  
2. 支持多种模型切换，包括通用模型、动漫模型和轻量模型。  
3. 支持原图与超分结果同步展示，并标注输入与输出分辨率。  
4. 支持结果图像放大查看，便于观察局部细节。  
5. 支持批量图像处理，降低重复操作成本。  
6. 支持单张结果下载，以及批量结果和批量对比图的 ZIP 一键下载。  
7. 支持标准化评测，输出可直接用于论文整理的指标表格与对比图表。

#### 3.1.2 非功能需求

1. 易用性：界面布局直观，操作流程简洁。  
2. 可扩展性：预留接入新模型和新评价指标的接口。  
3. 可维护性：前端、后端和评测模块之间保持相对解耦。  
4. 稳定性：在批量处理、文件下载和模型缓存等操作中避免重复加载或数据丢失。  
5. 实用性：既能够直观展示算法效果，也能够服务于论文实验分析。

### 3.2 系统总体架构设计

系统主要由四个模块组成：

1. 前端交互模块：负责图像上传、模型选择、结果展示、对比图下载与弹窗预览。  
2. 后端推理模块：负责模型初始化、权重加载、超分推理与结果编码。  
3. 批处理管理模块：负责多文件输入、结果缓存、文件保存与 ZIP 打包下载。  
4. 实验评测模块：负责测试集读取、低分辨率图像生成、多模型对比、指标计算与图表输出。

系统整体工作流程如下：用户在前端上传图像并选择模型，前端以 `multipart/form-data` 方式将图像发送至 `/process` 接口；后端完成请求解析与模型推理后，返回超分结果、拼接对比图以及尺寸信息；前端据此完成可视化展示与下载操作。对于批量处理任务，系统依次处理多张输入图像，并将结果及对比图缓存在本地目录与内存中，以便后续打包下载。

### 3.3 数据与目录组织设计

| 目录或文件 | 功能说明 |
| --- | --- |
| `web/index.html` | 前端页面结构与样式 |
| `web/script.js` | 前端交互逻辑、上传请求与结果下载 |
| `web/web_server.py` | 后端 HTTP 服务、模型封装与批处理接口 |
| `realesrgan/` | Real-ESRGAN 网络及推理代码 |
| `weights/` | 模型权重与 LPIPS 缓存文件 |
| `datasets/` | 测试集图像及自动生成的 `LR_bicubic/X4` |
| `evaluate.py` | 多模型评测与指标计算脚本 |
| `evaluation_results/` | 实验输出结果、图表和统计文件 |

### 3.4 系统业务流程设计

单图像超分业务流程为：图像上传 → 前端预览 → 模型选择 → 后端推理 → 结果返回 → 结果展示与下载。  
批量处理业务流程为：多图选择 → 批量提交 → 逐张推理 → 结果缓存 → 单张下载与 ZIP 批量下载。  
评测业务流程为：读取 HR 图像 → 生成或复用 LR 图像 → 模型推理 → 计算 PSNR、SSIM 和 LPIPS → 保存 SR 图像与对比图 → 导出统计表和柱状图。

## 第4章 系统设计与实现

### 4.1 开发环境与运行环境

系统开发环境为 Windows、Python 3.10、PyTorch 和 OpenCV，前端采用原生 HTML、CSS 和 JavaScript 实现，后端基于 Python 标准库构建轻量级 HTTP 服务。模型推理依赖 `RealESRGANer` 封装器，图像评测基于 `basicsr.metrics` 中的 PSNR 和 SSIM 计算函数，并结合 `lpips` 库计算 LPIPS 指标。整体环境配置较为简洁，适合在个人计算机上部署与演示。

### 4.2 模型推理模块实现

后端推理模块位于 `web/web_server.py`。为避免模型在多次请求中被重复加载，系统采用全局缓存机制保存已初始化的模型实例。当用户选择 `RealESRGAN_x4plus`、`RealESRGAN_x4plus_anime_6B` 或 `realesr-general-x4v3` 时，后端根据模型类型构造相应网络结构，并从 `weights` 目录读取预训练权重；若本地不存在对应权重文件，则系统尝试自动下载。模型完成加载后，统一通过 `RealESRGANer` 实现图像增强推理。

在具体处理流程上，上传图像首先经 `cv2.imdecode` 解码为矩阵形式，随后送入超分辨率模型进行推理，得到的结果再经 `cv2.imencode` 编码为 PNG 格式，并以 Base64 形式返回前端。该实现方式减少了中间文件读写过程，使前后端交互更加直接和高效。

### 4.3 单图像处理功能实现

单图像处理功能由前端页面与后端推理接口协同完成。用户选择图片后，前端通过 `FileReader` 完成原图预览，并构造 `FormData` 请求发送至 `/process` 接口。后端完成推理后，返回超分结果图、拼接对比图、输入尺寸、输出尺寸以及模型名称等信息，前端据此更新页面状态并显示处理结果。

在界面设计方面，系统采用左右双栏布局展示原图与超分结果，使用户能够直观比较前后差异。原图下载按钮设置于左侧区域，超分结果和对比图下载按钮设置于右侧区域，从而使界面逻辑更加清晰，也有利于用户进行细节观察与结果保存。

### 4.4 批量处理与 ZIP 下载功能实现

批量处理功能是本系统的重要扩展内容。用户点击“批量处理”后，可一次性选择多张图片，并通过“开始批量超分”将其统一提交至 `/batch_process` 接口。后端在接收到批量请求后，依次完成图像解码、模型推理、结果保存和对比图生成，并将每张图像的处理结果组织后返回前端。与此同时，系统将批处理结果缓存在内存与本地目录 `batch_outputs/<batch_id>/` 中，以支持后续下载操作。

针对批量结果下载需求，系统进一步设计了 `/batch_download_zip/<batch_id>/output` 和 `/batch_download_zip/<batch_id>/compare` 两类接口。当前端发起下载请求时，后端优先从内存缓存中读取结果数据，并利用 `BytesIO` 和 `zipfile` 在内存中动态构建 ZIP 压缩包，再以流式方式返回浏览器。该方案减少了磁盘读写开销，同时有效规避了中文路径环境下可能出现的兼容性问题，提升了系统在批量导出场景中的稳定性。

### 4.5 对比图与可视化展示实现

为了便于分析超分效果，系统提供了结果放大查看与对比图下载功能。后端通过对原图与结果图进行统一排版，生成包含标题与尺寸信息的拼接对比图。前端则借助模态框组件实现大图弹窗预览，用户可通过点击原图或结果图查看局部细节。该设计不仅增强了系统的交互体验，也便于实验结果在论文中的整理与展示。

### 4.6 评测模块实现

实验评测模块由 `evaluate.py` 实现，其主要流程包括：首先读取高分辨率测试图像；然后对图像进行 `mod_crop` 处理，以保证尺寸可被放大倍率整除；接着采用双三次下采样生成对应低分辨率图像，并保存至指定目录；随后调用不同模型对低分辨率输入进行超分辨率重建；最后计算 PSNR、SSIM 和 LPIPS 等评价指标，并导出逐图像结果及统计文件。

该评测模块支持 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3 四种模型。为保证 LPIPS 在本地环境中的稳定计算，系统对感知模型权重的缓存路径进行了统一处理，从而避免因运行环境差异导致的下载或读写问题。通过该模块，系统形成了从测试输入生成、模型推理、指标计算到结果导出的完整实验闭环。

### 4.7 关键实现说明

系统实现中，模型初始化、单图像处理、批量任务调度、压缩包导出、低分辨率测试图像生成以及多模型评测等功能分别进行了模块化封装。具体而言，后端通过统一的模型初始化机制完成不同网络结构与预训练权重的匹配，并结合缓存策略减少重复加载开销；图像处理流程被抽象为可复用的推理接口，从而为单图像处理与批量处理提供统一支撑；批量任务管理模块负责组织多图输入、结果缓存与 ZIP 导出；评测部分则实现了测试数据准备、逐模型推理、指标统计与结果写出。上述设计保证了系统在功能扩展、实验复现和代码维护方面具有较好的可操作性。

通过上述实现，本文不仅完成了超分辨率系统原型的构建，也形成了较为完整的评测与分析流程。

## 第5章 系统测试与实验结果分析

### 5.1 实验目的与实验环境

本章实验主要验证以下三个方面：其一，系统能否稳定完成超分辨率重建与结果展示；其二，不同模型在公开数据集上的客观失真指标与感知质量指标表现；其三，本文扩展实现的批处理、结果导出和图表生成功能能否满足论文分析需求。

实验在本地 Windows 平台上完成。硬件环境为 Intel Core i7-12700H 处理器、NVIDIA GeForce RTX 3060 Laptop 显卡（6 GB 显存），操作系统为 Windows 11。软件环境包括 Python 3.10、PyTorch 1.13.1、CUDA 11.7 和 lpips 0.1.4。所有模型推理均在 GPU 半精度（FP16）模式下进行。实验脚本为 `evaluate.py`，绘图脚本为 `plot_evaluation_results.py`。

### 5.2 数据集与测试方案

#### 5.2.1 测试数据集

本文选用 Set5、Set14 和 BSD100 三个公开数据集进行实验。这三组数据集涵盖人物、动物、自然纹理及复杂场景，是图像超分辨率研究中常用的标准测试集。测试过程中，`datasets/<dataset>/HR/` 用于存放高分辨率 Ground Truth 图像，并以其为基准经双三次下采样生成对应低分辨率输入，保存至 `LR_bicubic/X4` 路径。所有模型共用同一组低分辨率输入，以保证实验的公平性与可比性。

#### 5.2.2 对比模型

本文选取以下四种方法进行比较：

1. Bicubic：双三次插值基线，用于反映传统插值方法的性能。  
2. ESRGAN_x4：经典感知型 GAN 超分辨率模型。  
3. RealESRGAN_x4plus：Real-ESRGAN 通用模型。  
4. realesr-general-x4v3：轻量化 Real-ESRGAN 通用模型。

#### 5.2.3 评价指标

实验采用 PSNR、SSIM 和 LPIPS 三项指标，从像素保真度、结构相似性和感知质量三个维度对模型进行评估。实验放大倍率固定为 ×4，`crop_border = 4`，并在 Y 通道上计算 PSNR 和 SSIM。

### 5.3 定量实验结果

#### 5.3.1 Set5 数据集结果

表5-1 不同模型在 Set5 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 27.6918 | 0.8102 | 0.3470 |
| ESRGAN_x4 | 17.2647 | 0.3670 | 0.5345 |
| RealESRGAN_x4plus | 25.4259 | 0.7851 | 0.1718 |
| realesr-general-x4v3 | 25.5326 | 0.7933 | 0.1806 |

由表5-1可以看出，Bicubic 在 PSNR 和 SSIM 上取得最高数值，而两种 Real-ESRGAN 模型在 LPIPS 上明显更优，说明其在感知质量方面显著优于传统插值方法和 ESRGAN_x4。其中，realesr-general-x4v3 的 PSNR 和 SSIM 略高于 RealESRGAN_x4plus，表明轻量模型在该数据集上的结构保真度表现略占优势。

#### 5.3.2 Set14 数据集结果

表5-2 不同模型在 Set14 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 25.0055 | 0.7054 | 0.4603 |
| ESRGAN_x4 | 15.7702 | 0.2534 | 0.6229 |
| RealESRGAN_x4plus | 23.6790 | 0.6678 | 0.2453 |
| realesr-general-x4v3 | 23.7539 | 0.6827 | 0.2564 |

在 Set14 数据集上，整体趋势与 Set5 基本一致。两种 Real-ESRGAN 模型的 LPIPS 远优于 Bicubic 和 ESRGAN_x4，说明其在视觉感知质量上具有明显优势。同时，RealESRGAN_x4plus 的 LPIPS 更低，而 realesr-general-x4v3 在 PSNR 和 SSIM 上略优，体现出两种模型在感知质量与结构保真度之间的不同侧重。

#### 5.3.3 BSD100 数据集结果

表5-3 不同模型在 BSD100 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 24.6329 | 0.6648 | 0.5285 |
| ESRGAN_x4 | 15.9418 | 0.2670 | 0.6021 |
| RealESRGAN_x4plus | 23.4753 | 0.6194 | 0.2858 |
| realesr-general-x4v3 | 23.7871 | 0.6369 | 0.3083 |

在 BSD100 数据集上，realesr-general-x4v3 在 PSNR 和 SSIM 上略高于 RealESRGAN_x4plus，而后者在 LPIPS 上更低。该结果说明两者在保真度与感知质量上的表现具有一定互补性。

### 5.4 实验结果综合分析

#### 5.4.1 失真指标与感知指标的权衡

Bicubic 在三组数据集上的 PSNR 均取得最高值，主要原因在于本文测试所用的低分辨率图像由双三次下采样生成，因此 Bicubic 插值与测试降质过程具有较高一致性，从而在像素级误差上占优。然而，从 LPIPS 指标来看，Bicubic 的结果明显劣于 Real-ESRGAN，表明其重建图像虽然在数值上更接近参考图像，但视觉上往往更加平滑、纹理不足。这一现象与感知—失真权衡理论[9]一致，即较低的失真并不必然对应更优的感知质量。

#### 5.4.2 LPIPS 指标的指导意义

实验结果表明，RealESRGAN_x4plus 与 realesr-general-x4v3 在三组数据集上的 LPIPS 均明显优于 Bicubic 和 ESRGAN_x4。由此可见，仅依赖 PSNR 和 SSIM 难以全面反映真实场景超分模型的优势，而引入 LPIPS 有助于更准确地评价模型在纹理真实性和视觉观感方面的表现。对于以感知质量为目标的超分辨率模型而言，LPIPS 具有重要的参考价值。

#### 5.4.3 两种 Real-ESRGAN 模型的差异

实验结果显示，realesr-general-x4v3 在 PSNR 和 SSIM 上略占优势，而 RealESRGAN_x4plus 的 LPIPS 更低。这表明轻量模型在结构保真方面具有一定优势，而通用模型在感知质量方面表现更佳。该差异与模型结构设计及训练目标相一致，也说明在不同应用场景下，可以根据实际需求在保真度与感知质量之间进行灵活选择。

#### 5.4.4 ESRGAN_x4 指标偏低的原因

ESRGAN_x4 在三组数据集上的各项指标均明显偏低，反映出其与本文采用的标准双三次降质测试协议匹配度不足。相较于 Real-ESRGAN，ESRGAN 对复杂退化与真实场景分布的适应能力较弱，在测试中更易出现纹理伪影或结构偏差。因此，本文将其作为历史基线方法进行参考比较。

### 5.5 图表结果说明

本文生成了三张可直接用于论文排版的柱状图，其图题分别为：

图5-1 不同模型在 Set5、Set14 和 BSD100 上的 PSNR 对比  
图5-2 不同模型在 Set5、Set14 和 BSD100 上的 SSIM 对比  
图5-3 不同模型在 Set5、Set14 和 BSD100 上的 LPIPS 对比（Lower Better）

图中蓝、橙、绿、紫四种颜色分别对应 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3，各图例与正文表述保持一致。通过柱状图可以更加直观地观察到：Bicubic 在 PSNR 和 SSIM 上占优，而 Real-ESRGAN 两模型在 LPIPS 上显著领先，这与前述定量分析结论一致。

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

测试结果表明，系统已完整覆盖毕业设计任务书中的主要功能需求，能够满足图像超分辨率处理、结果展示、批量导出和实验评测等应用场景。

### 5.7 本章小结

本章从数据集与测试方案、定量结果、图表输出以及系统功能测试等方面对本文所实现的系统进行了验证。实验结果表明，系统能够稳定完成超分辨率重建与多模型评测任务；Real-ESRGAN 模型在 LPIPS 指标上明显优于传统插值方法和上一代 GAN 方法；系统前后端交互、批量处理和结果导出功能运行稳定，达到了毕业设计中“系统原型实现与实验验证”的目标。

## 第6章 总结与展望

### 6.1 工作总结

本文围绕“基于 Real-ESRGAN 的图像超分辨率系统设计与实现”展开研究。首先，梳理了图像超分辨率相关理论，分析了 Real-ESRGAN 的生成器结构、判别器设计、高阶降质建模与损失函数；其次，基于官方公开预训练模型完成了后端推理模块与 Web 可视化界面的构建，并实现了单图像处理、批量处理、对比展示、细节放大和 ZIP 批量导出等功能；最后，扩展实现了评测模块，支持多模型比较以及 PSNR、SSIM、LPIPS 联合评估。

### 6.2 主要成果与创新点

1. 构建了从模型调用到界面交互的完整超分辨率系统，实现了单图像与批量图像处理全流程。  
2. 在系统中实现了批量结果和批量对比图的 ZIP 一键下载功能，增强了工程实用性。  
3. 设计并实现了多模型评测脚本，引入 LPIPS 指标，将分析维度由传统保真度评价拓展到感知质量评价。  
4. 在 Set5、Set14 和 BSD100 数据集上完成四模型对比实验，量化展示了感知—失真权衡特征。  
5. 形成了一套与项目代码一致的论文实验流程，可为毕业答辩展示及后续优化提供支撑。

### 6.3 不足与展望

尽管本文完成了系统设计、实现与实验验证，但仍存在以下不足：

1. 未针对特定数据集进行微调，因此在标准双三次退化测试集上的 PSNR 和 SSIM 仍有进一步提升空间。  
2. 系统主要面向本地单机部署，尚未扩展为支持多用户并发访问的服务形态。  
3. 当前评测数据集仅覆盖 Set5、Set14 和 BSD100，测试范围仍相对有限。  
4. 系统尚未集成用户管理、历史记录和任务队列等完整应用功能。

未来可从以下几个方向继续开展研究：  
（1）面向双三次退化或特定真实场景开展模型微调，以提升 PSNR 和 SSIM 等保真度指标；  
（2）引入扩散模型等新型真实场景超分辨率方法，进一步改善图像感知质量；  
（3）优化系统部署架构，引入更完善的后端框架以提高并发处理能力与运行效率；  
（4）增加运行时间、显存占用和主观评分等多维度评价指标，构建更加全面的系统评估体系。

## 参考文献

[1] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision. Cham: Springer, 2014: 184-199.

[2] Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 4681-4690.

[3] Wang X, Yu K, Wu S, et al. ESRGAN: Enhanced super-resolution generative adversarial networks[C]//Proceedings of the European Conference on Computer Vision Workshops. 2018: 0-0.

[4] Johnson J, Alahi A, Li F F. Perceptual losses for real-time style transfer and super-resolution[C]//European Conference on Computer Vision. Cham: Springer, 2016: 694-711.

[5] Wang X, Xie L, Dong C, et al. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. 2021: 1905-1914.

[6] 刘东, 董超, 王兴刚, 等. Real-ESRGAN：真实世界图像超分辨率研究[J]. 计算机研究与发展, 2021.

[7] 唐艳秋, 潘泓, 朱亚平, 等. 图像超分辨率重建研究综述[J]. 电子学报, 2020, 48(7): 1407-1420.

[8] 王睿琪. 图像超分辨率重建综述[J]. 计算机科学与应用, 2024, 14(2): 350-359.

[9] Blau Y, Michaeli T. The perception-distortion tradeoff[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 6228-6237.

[10] Zhang R, Isola P, Efros A A, et al. The unreasonable effectiveness of deep features as a perceptual metric[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 586-595.

[11] Horé A, Ziou D. Image quality metrics: PSNR vs. SSIM[C]//Proceedings of the International Conference on Pattern Recognition. 2010: 2366-2369.

[12] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.

## 致谢

在本次毕业设计与论文撰写过程中，我系统梳理了图像超分辨率相关理论，并结合项目代码完成了 Real-ESRGAN 超分辨率系统的设计、实现与测试。在此，谨向指导教师在课题选题、系统实现和论文修改过程中给予的耐心指导与帮助表示衷心感谢。感谢开源社区提供 Real-ESRGAN 相关基础代码与公开预训练权重，感谢公开数据集和相关论文作者为模型测试与性能分析提供了重要参考。通过本次毕业设计，我进一步加深了对图像超分辨率与深度学习方法的理解，并在系统集成、实验评测和学术写作等方面获得了较为全面的锻炼。

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

[# 基于Real-ESRGAN的图像超分辨率系统设计与实现

> 排版说明  
> 1. 首页、诚信承诺书、版权声明等前置页面按学校模板单独填写。  
> 2. 正文中的英文、数字、变量名、文件名、模型名统一采用 `Times New Roman`。  
> 3. 文中公式建议在 Word 中使用“公式编辑器”重新录入。  
> 4. 目录、图目录、表目录建议使用 Word 自动生成。  
> 5. 本稿以你当前项目代码、系统功能和已完成实验结果为基础撰写，可直接作为论文正文初稿继续细化。

## 摘要

图像超分辨率重建旨在从低分辨率图像中恢复高分辨率细节，是计算机视觉与图像处理领域的重要研究方向，在老旧图像修复、视频监控增强、移动终端成像优化和数字内容修复等场景中具有较高应用价值。传统插值方法虽然实现简单、计算开销低，但在复杂退化条件下往往难以有效恢复纹理细节，容易出现模糊、锯齿和边缘失真等问题。随着深度学习方法的发展，以生成对抗网络为代表的感知驱动型超分辨率模型在视觉质量方面取得了显著进展，其中 Real-ESRGAN 通过高阶退化建模和改进判别器设计，提升了模型对真实场景低质图像的适应能力。

针对传统方法在真实场景中泛化能力不足、交互性较差以及工程落地不便的问题，本文设计并实现了一套基于 Real-ESRGAN 的图像超分辨率系统。系统以后端推理服务为核心，采用 Python、PyTorch 和 OpenCV 完成模型加载、图像预处理与超分辨率重建，采用 Web 技术构建可视化交互界面，实现了单图像超分、批量处理、原图与超分结果同步对比、放大查看以及结果下载等功能。同时，为增强实验分析的完整性，本文扩展实现了评测模块，支持 Bicubic、ESRGAN、RealESRGAN_x4plus 和 realesr-general-x4v3 等多模型对比，并引入 PSNR、SSIM 和 LPIPS 三类指标对系统性能进行综合评价。

在公开数据集 Set5、Set14 和 BSD100 上的实验结果表明，所设计系统能够稳定完成图像超分辨率重建任务。以 `realesr-general-x4v3` 模型为例，其在 Set5、Set14 和 BSD100 数据集上的平均 PSNR 分别达到 `25.5326 dB`、`23.7539 dB` 和 `23.7871 dB`，平均 SSIM 分别为 `0.7933`、`0.6827` 和 `0.6369`；`RealESRGAN_x4plus` 在三组数据上的平均 LPIPS 分别为 `0.1718`、`0.2453` 和 `0.2858`，表现出较好的感知质量。实验结果说明，本文构建的系统在真实场景图像增强任务中具有良好的可用性和一定的工程应用价值。

关键词：图像超分辨率；Real-ESRGAN；生成对抗网络；Web系统；图像增强

## Abstract

Image super-resolution reconstruction aims to recover high-resolution details from low-resolution inputs and has important research and application value in image restoration, surveillance enhancement, mobile imaging, and digital content repair. Traditional interpolation-based methods are easy to implement and computationally efficient, but they usually fail to recover rich textures under complex degradations, often producing blurry edges and visual artifacts. With the rapid development of deep learning, perceptual super-resolution methods based on generative adversarial networks have achieved remarkable progress, and Real-ESRGAN has become a representative real-world blind super-resolution model due to its high-order degradation modeling strategy and improved discriminator design.

To address the limitations of traditional methods in real-world generalization, usability, and engineering deployment, this thesis designs and implements an image super-resolution system based on Real-ESRGAN. The system takes a backend inference service as the core and uses Python, PyTorch, and OpenCV to accomplish model loading, image preprocessing, and super-resolution reconstruction. A Web-based interactive interface is further developed to support single-image enhancement, batch processing, synchronized comparison between input and output images, detail zooming, and result downloading. In addition, an evaluation module is implemented to support multi-model comparison among Bicubic, ESRGAN, RealESRGAN_x4plus, and realesr-general-x4v3, and three metrics, namely PSNR, SSIM, and LPIPS, are adopted for comprehensive performance assessment.

Experimental results on the public datasets Set5, Set14, and BSD100 show that the proposed system can reliably accomplish image super-resolution tasks. For example, the `realesr-general-x4v3` model achieves average PSNR values of `25.5326 dB`, `23.7539 dB`, and `23.7871 dB` on Set5, Set14, and BSD100, respectively, with average SSIM values of `0.7933`, `0.6827`, and `0.6369`. Meanwhile, `RealESRGAN_x4plus` achieves average LPIPS values of `0.1718`, `0.2453`, and `0.2858` on the three datasets, indicating favorable perceptual quality. The results demonstrate that the designed system is effective, practical, and valuable for real-world image enhancement applications.

Keywords: image super-resolution; Real-ESRGAN; generative adversarial network; Web system; image enhancement

## 第1章 绪论

### 1.1 研究背景与意义

#### 1.1.1 现实意义

随着数字图像在安防监控、智能终端、医学辅助诊断、数字档案修复和网络媒体传播中的广泛应用，图像质量已成为影响后续识别、分析和视觉体验的重要因素。由于采集设备性能限制、传输压缩、拍摄抖动和噪声干扰等原因，大量图像存在分辨率不足、纹理细节缺失和边缘模糊等问题。若直接依赖硬件升级解决这类问题，往往成本较高，且对历史数据、老旧图像和网络压缩图像无能为力。因此，利用软件算法提升低分辨率图像质量，具有明显的现实意义和应用价值。

图像超分辨率技术能够在不改变原始采集硬件的前提下提升图像清晰度，为后续的检测、识别、理解和展示提供更加丰富的细节信息。在实际场景中，该技术可用于修复旧照片、增强监控截图、提升移动端图像显示效果、辅助文本图像识别，并为图像修复和画质增强提供前置处理支持。尤其是在真实退化复杂、图像来源多样的应用环境下，构建一套既具备较好重建效果又便于交互使用的超分辨率系统，具有明显的工程意义。

#### 1.1.2 研究价值

从研究角度看，图像超分辨率问题本质上是一个病态逆问题，即一个低分辨率图像对应多个可能的高分辨率解。传统方法多依赖固定退化假设，往往只能在理想退化模型下取得较好效果，而面对真实场景中的模糊、噪声、压缩失真和未知退化时，常出现泛化能力不足的问题。近年来，深度学习特别是生成对抗网络的发展推动了超分辨率研究不断向真实场景、感知质量和工程应用方向演进[1-7]。

Real-ESRGAN 作为面向真实场景盲超分辨率的代表性模型，在保留 ESRGAN 感知增强能力的基础上，引入高阶退化建模和 U-Net 判别器结构，在真实图像恢复中显示出较好的适应性[12-13]。因此，围绕 Real-ESRGAN 开展系统设计与实现研究，不仅能够帮助理解真实场景超分辨率算法的关键机理，也有助于推进该类模型在实际应用系统中的落地。

### 1.2 国内外研究现状

#### 1.2.1 国外研究现状

国外对图像超分辨率的研究起步较早，早期方法主要集中在插值重建、稀疏表示和样本学习等方向。经典方法如双线性插值、双三次插值实现简单，但难以恢复高频细节；基于样本字典或稀疏表示的方法能够在一定程度上提升重建质量，但计算复杂度较高，且对先验依赖明显。随着深度学习的发展，超分辨率研究逐渐转向卷积神经网络方法。Dong 等提出 SRCNN，首次将卷积神经网络用于单幅图像超分辨率任务，为深度学习超分辨率研究奠定了基础[1]。随后，VDSR、EDSR 和 RCAN 等模型通过加深网络层数、引入残差结构和通道注意力机制，不断提升像素级重建性能[2-5]。

在追求更高 PSNR 和 SSIM 的同时，研究者逐渐认识到高客观指标并不一定意味着更好的主观视觉质量。为改善超分辨率图像的感知效果，Ledig 等提出 SRGAN，将生成对抗网络引入超分辨率任务，在纹理重建方面取得了突破[4]。Wang 等进一步提出 ESRGAN，通过改进残差块结构和感知损失设计，进一步增强了纹理表现能力[6]。此后，感知质量与失真指标之间的平衡问题成为该领域的重要研究主题。

随着真实场景图像恢复需求增加，研究重点进一步转向盲超分辨率和真实退化建模。Zhang 等提出更实用的退化模型，为盲超分辨率训练提供了更接近真实图像的合成数据基础[13]；Wang 等提出 Real-ESRGAN，将高阶退化建模和谱归一化 U-Net 判别器结合起来，在真实图像恢复中取得较好的视觉效果[12]。近年来，SwinIR、ATD-SR、CFAT 等模型将 Transformer 引入超分辨率任务，提高了建模全局上下文的能力[10,17,20]；SeeSR、AdcSR、Dual-LoRA 等方法则探索了语义先验、扩散模型和多模态信息在真实场景超分辨率中的应用[18,21-23]。由此可见，国外研究已从“提高像素级重建精度”逐步扩展到“兼顾感知质量、泛化能力与推理效率”的综合目标。

#### 1.2.2 国内研究现状

国内关于图像超分辨率的研究近年来发展迅速，研究内容涉及方法综述、网络结构改进、遥感与医学图像应用、真实场景图像修复等多个方面。唐艳秋等对图像超分辨率重建研究进行了系统综述，梳理了传统方法与深度学习方法的发展脉络[24]；王睿琪从单图像和多图像两个角度总结了超分辨率重建的发展现状与挑战[25]。这些研究表明，国内学者已充分关注超分辨率任务在理论模型、性能评价和行业应用方面的综合发展。

在算法研究方面，国内团队不仅关注模型深度与结构优化，也逐渐重视复杂退化建模、轻量化部署和实际系统应用。例如，围绕真实场景超分辨率、图像增强及图像修复的研究越来越多地采用开源预训练模型作为基础，在此之上开展应用适配、模块集成和评测扩展工作，以提升系统工程可用性。与国外研究相比，国内研究在系统化应用实现和场景适配方面具有较强实践导向，但在高质量公开数据集建设、统一评测协议以及真实复杂场景泛化能力研究方面仍有进一步提升空间。

### 1.3 现有研究存在的主要问题

综合国内外研究现状，可以发现当前图像超分辨率研究仍存在以下几个方面的问题。

第一，真实场景退化复杂，模型泛化能力不足。多数高指标模型默认低分辨率图像由标准双三次下采样得到，而真实场景中的模糊、噪声、压缩失真和传感器噪声往往是叠加存在的，导致模型在真实图像上容易出现过平滑、细节伪造或结构失真等问题。

第二，客观指标与主观感知存在矛盾。以 PSNR 和 SSIM 为代表的失真指标更偏向像素级一致性，而以 GAN 或扩散模型为代表的感知增强方法更重视纹理真实感，两者之间常常存在权衡关系。如何在保持结构真实性的同时提升视觉观感，仍然是实际应用中的关键问题。

第三，算法与系统脱节现象较为明显。许多研究重点聚焦模型本身，却忽略了文件上传、交互预览、批量处理、结果管理和一键下载等工程功能，导致算法难以直接服务于实际应用。

第四，评测流程不够完整。部分系统仅展示视觉结果，缺少标准化测试集、统一退化方式以及多指标评价机制，不利于对模型性能进行客观比较和复现验证。

### 1.4 本文主要研究内容

围绕上述问题，本文结合当前项目实现，完成了如下研究内容。

1. 对图像超分辨率及 Real-ESRGAN 的关键理论进行梳理，重点分析生成器、判别器、高阶退化建模和损失函数设计。

2. 基于 Python、PyTorch 和 OpenCV 封装 Real-ESRGAN 推理流程，构建面向实际使用的图像超分辨率后端服务。

3. 采用 Web 方式实现图像超分辨率系统前端界面，支持模型选择、单图像超分、结果对比、细节查看、批量处理以及 ZIP 一键下载等功能。

4. 扩展实现模型评测脚本，支持自动生成 `LR_bicubic/X4` 测试输入，支持 Bicubic、ESRGAN、RealESRGAN_x4plus 和 realesr-general-x4v3 多模型比较，并引入 PSNR、SSIM 和 LPIPS 三项指标。

5. 在公开数据集 Set5、Set14 和 BSD100 上完成系统测试与实验分析，讨论不同模型在失真指标与感知指标上的差异，并总结当前系统的优点与不足。

需要说明的是，本文的研究重点在于“系统设计与实现”及“模型应用验证”。受硬件资源与毕业设计周期限制，本文未对 Real-ESRGAN 进行从头训练，而是基于官方公开预训练权重完成模型集成、接口封装、评测扩展和可视化系统实现。这一处理方式与当前项目代码和实验结果保持一致。

### 1.5 论文组织结构

全文共分为六章。

第1章为绪论，介绍课题研究背景、现实意义、国内外研究现状、现有问题以及本文的研究内容。  
第2章介绍图像超分辨率与 Real-ESRGAN 的相关理论基础，包括模型结构、退化建模、损失函数和评价指标。  
第3章对系统需求进行分析，并给出整体架构与功能设计。  
第4章详细说明系统的关键实现过程，包括前后端交互、模型推理封装、批量处理和评测模块。  
第5章给出实验设计、测试流程、实验结果与分析。  
第6章总结全文工作，并对后续研究方向进行展望。

## 第2章 相关理论与关键技术

### 2.1 图像超分辨率基本概念

图像超分辨率（Super-Resolution，SR）是指从低分辨率图像中恢复高分辨率图像的过程。按照输入图像数量不同，可分为单幅图像超分辨率和多幅图像超分辨率；按照退化是否已知，可分为非盲超分辨率和盲超分辨率。本文研究对象为单幅图像盲超分辨率，即仅给定一幅低分辨率图像，在未知退化条件下恢复较高分辨率图像。

在数学上，低分辨率图像通常可表示为高分辨率图像经过模糊、下采样、噪声和压缩等退化过程后的结果。设高分辨率图像为 $I_{HR}$，低分辨率图像为 $I_{LR}$，则退化过程可抽象为：

$$
I_{LR}=D(I_{HR};\theta)+n
$$

其中，$D(\cdot)$ 表示退化函数，$\theta$ 表示退化参数，$n$ 表示噪声项。超分辨率的目标是学习一个映射函数 $F(\cdot)$，使得：

$$
\hat{I}_{HR}=F(I_{LR})
$$

由于多个高分辨率图像可能对应同一个低分辨率观测结果，因此该问题具有典型的不适定性。

### 2.2 Real-ESRGAN 模型原理

#### 2.2.1 生成器结构

Real-ESRGAN 的生成器继承了 ESRGAN 中的 RRDB（Residual-in-Residual Dense Block）结构。RRDB 通过残差连接与稠密连接相结合，在不引入批归一化层的前提下增强了特征复用能力，既能提高网络表达能力，又有助于稳定训练过程[6,12]。与传统卷积网络相比，RRDB 在纹理细节恢复方面更具优势，适合用于复杂图像重建任务。

在工程实现中，`RealESRGAN_x4plus` 模型采用 23 个 RRDB 模块，`RealESRGAN_x4plus_anime_6B` 采用较浅的 6 个 RRDB 模块，而 `realesr-general-x4v3` 则采用更轻量的 `SRVGGNetCompact` 结构，以兼顾推理速度和部署成本。

#### 2.2.2 高阶退化建模

Real-ESRGAN 的一个重要贡献在于提出了更加贴近真实场景的高阶退化建模策略[12-13]。相比仅使用单次模糊和双三次下采样的传统做法，该方法通过多轮随机模糊、缩放、噪声扰动和 JPEG 压缩，模拟真实世界图像在采集、存储和传播过程中可能经历的复杂退化。其核心思想是利用更加丰富的合成退化过程缩小训练数据与真实图像之间的分布差异，从而提升模型对真实输入的适应能力。

该思想对本文系统具有直接指导意义。虽然本文未重新训练模型，但在理解 Real-ESRGAN 适用场景和实验结果时，高阶退化建模是解释其为何优于传统感知超分模型的重要理论基础。

#### 2.2.3 判别器结构

ESRGAN 主要采用基于 PatchGAN 思想的判别器，而 Real-ESRGAN 进一步引入带谱归一化的 U-Net 判别器[12]。这种结构兼顾全局判别和局部判别能力，有助于提升生成结果的结构一致性和纹理真实感。谱归一化能够限制判别器参数增长，提高训练稳定性，减少训练过程中梯度爆炸和不稳定振荡的问题。

#### 2.2.4 损失函数设计

Real-ESRGAN 通过内容损失、感知损失与对抗损失的联合优化来平衡重建精度与感知质量。其总损失可表示为：

$$
\mathcal{L}_{total}=\lambda_1\mathcal{L}_{1}+\lambda_p\mathcal{L}_{percep}+\lambda_g\mathcal{L}_{GAN}
$$

其中，$\mathcal{L}_{1}$ 用于约束输出结果与目标图像在像素空间上的差异；$\mathcal{L}_{percep}$ 通常基于预训练分类网络特征提取，强调语义结构和高层感知一致性；$\mathcal{L}_{GAN}$ 则推动生成器输出更具真实纹理的结果。三者协同优化，使模型既能保持整体结构，又能恢复更自然的高频细节。

### 2.3 图像质量评价指标

为了从不同角度评估超分辨率图像质量，本文采用 PSNR、SSIM 和 LPIPS 三种指标。

#### 2.3.1 PSNR

峰值信噪比（Peak Signal-to-Noise Ratio，PSNR）常用于衡量重建图像与参考图像之间的像素级误差。其计算依赖均方误差（Mean Squared Error，MSE）：

$$
MSE=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\left(I_{HR}(i,j)-\hat{I}_{HR}(i,j)\right)^2
$$

则 PSNR 为：

$$
PSNR=10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

其中，$MAX_I$ 为图像像素最大值，8 位图像中通常取 255。PSNR 越高，表明重建结果与参考图像越接近。

#### 2.3.2 SSIM

结构相似性（Structural Similarity Index，SSIM）从亮度、对比度和结构三方面衡量图像相似性，较 PSNR 更贴近人类视觉感知。其定义为：

$$
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

其中，$\mu_x$、$\mu_y$ 分别表示两幅图像的均值，$\sigma_x^2$、$\sigma_y^2$ 表示方差，$\sigma_{xy}$ 表示协方差，$C_1$ 和 $C_2$ 为稳定常数。SSIM 取值越接近 1，表示结构越相似。

#### 2.3.3 LPIPS

学习感知图像块相似度（Learned Perceptual Image Patch Similarity，LPIPS）利用深度特征空间中的距离衡量两幅图像的感知差异[7]。与 PSNR 和 SSIM 偏重像素级或结构级一致性不同，LPIPS 更关注视觉感知层面的相似性。该指标数值越低，表示重建图像与参考图像在感知上越接近。由于 GAN 类模型常在感知质量方面优于简单插值方法，因此 LPIPS 对于分析“客观指标与主观质量权衡”具有重要意义。

### 2.4 Web 系统相关技术

本文系统采用前后端分离程度较低的轻量级 Web 架构实现。前端使用 `HTML + CSS + JavaScript` 构建交互界面，负责文件选择、模型切换、结果展示、放大预览以及批量结果下载；后端使用 Python 标准库中的 `http.server` 与 `socketserver` 构建简易服务，结合 `PyTorch` 完成模型推理，结合 `OpenCV` 完成图像解码、编码和结果拼接。该技术路线实现成本较低、依赖简单、便于本地部署，符合毕业设计系统原型实现的目标。

## 第3章 系统需求分析与总体设计

### 3.1 系统需求分析

#### 3.1.1 功能需求

结合毕业设计任务书和当前系统实现，本文系统应满足以下功能需求。

1. 支持用户上传单张图像进行超分辨率重建。  
2. 支持多种模型切换，包括通用模型、动漫模型和轻量模型。  
3. 支持原图与超分结果同步展示，并标注输入输出分辨率。  
4. 支持结果图片放大查看，便于观察局部细节。  
5. 支持批量图像处理，减少重复操作成本。  
6. 支持单张结果下载，以及批量结果和批量对比图的 ZIP 一键下载。  
7. 支持公开测试集评测，输出可用于论文分析的图表和指标结果。

#### 3.1.2 非功能需求

除功能需求外，系统还应满足以下非功能需求。

1. 易用性：界面布局清晰，操作步骤简洁，适合非专业用户使用。  
2. 可扩展性：后续可继续接入更多模型或新的评价指标。  
3. 可维护性：前端、后端、评测模块分离，便于代码维护。  
4. 稳定性：对于批量处理、文件下载和模型缓存等功能，应避免重复加载和数据丢失。  
5. 实用性：系统既能展示算法效果，也能服务于论文实验分析。

### 3.2 系统总体架构设计

本文系统总体上可分为四个模块：前端交互模块、后端推理模块、批处理管理模块和实验评测模块。

1. 前端交互模块负责图像上传、模型选择、结果展示、下载和弹窗预览。  
2. 后端推理模块负责模型初始化、权重加载、图像超分辨率推理和结果编码。  
3. 批处理管理模块负责多文件输入、处理结果缓存、结果文件保存和 ZIP 打包下载。  
4. 实验评测模块负责测试集读取、低分辨率图像生成、多模型对比、指标计算和结果绘图。

系统整体工作流程如下：用户在前端界面上传图像并选择模型，前端将图像以 `multipart/form-data` 方式提交至后端接口；后端解析图像数据并调用 Real-ESRGAN 推理模块生成超分结果；随后将结果图像、对比图像和尺寸信息返回前端，前端完成可视化展示与下载。若用户选择批量处理，则系统循环处理多张输入图像，并将超分结果和对比图缓存在本地目录及内存中，以便后续 ZIP 打包。

### 3.3 数据与目录组织设计

结合当前项目结构，系统相关目录组织如下。

| 目录或文件 | 功能说明 |
| --- | --- |
| `web/index.html` | 前端页面结构与样式 |
| `web/script.js` | 前端交互逻辑、上传请求、结果下载 |
| `web/web_server.py` | 后端 HTTP 服务、模型封装、批处理接口 |
| `realesrgan/` | Real-ESRGAN 相关网络与推理代码 |
| `weights/` | 模型权重与 LPIPS 缓存文件 |
| `datasets/` | 测试集图像与自动生成的 `LR_bicubic/X4` |
| `evaluate.py` | 多模型评测与指标计算脚本 |
| `evaluation_results/` | 实验输出结果、图表和统计文件 |

这种组织方式清晰地区分了系统运行模块和实验评测模块，有利于系统展示和论文实验分析同步开展。

### 3.4 系统业务流程设计

单图像超分业务流程包括：图像上传、前端预览、模型选择、后端推理、结果返回、结果展示与下载。  
批量处理业务流程包括：多图选择、批量提交、逐张推理、结果缓存、单张下载和 ZIP 批量下载。  
评测业务流程包括：读取 HR 图像、生成或复用 LR 图像、模型推理、计算指标、保存 SR 图像与可视化对比图、导出统计表和柱状图。

## 第4章 系统设计与实现

### 4.1 开发环境与运行环境

本文系统采用 `Windows + Python 3.10 + PyTorch + OpenCV` 的开发环境进行实现，前端采用原生 `HTML/CSS/JavaScript`，后端使用 Python 标准库搭建轻量级 HTTP 服务。模型推理依赖 `RealESRGANer` 封装器，图像评测依赖 `basicsr.metrics` 中的 PSNR 与 SSIM 计算函数，并使用 `lpips` 库扩展感知指标。整体环境配置简洁，便于在个人计算机上部署与演示。

### 4.2 模型推理模块实现

后端推理模块位于 `web/web_server.py`。为减少重复加载带来的时间开销，系统使用全局字典 `MODEL_CACHE` 对已初始化模型进行缓存。当用户选择 `RealESRGAN_x4plus`、`RealESRGAN_x4plus_anime_6B` 或 `realesr-general-x4v3` 时，后端根据模型名称构造对应网络结构，并从 `weights` 目录中读取权重；若本地不存在权重，则尝试自动下载。随后，通过 `RealESRGANer` 完成图像增强推理。

图像推理的核心流程为：读取上传文件、通过 `cv2.imdecode` 解码为矩阵、送入 `upsampler.enhance()` 进行超分辨率处理、再通过 `cv2.imencode` 编码为 PNG，并以 `Base64` 形式返回前端。该实现方式避免了复杂的文件传输过程，使前后端交互更加直接。

### 4.3 单图像处理功能实现

单图像处理功能的前端实现位于 `web/script.js`。用户选择图片后，前端首先通过 `FileReader` 完成原图预览，随后构造 `FormData` 请求并发送至 `/process` 接口。后端完成推理后，会返回超分结果图、拼接对比图、输入尺寸、输出尺寸及模型名称。前端在收到响应后将结果展示在“超分结果”区域，并更新尺寸信息和状态提示。

为提高可视化效果，系统采用左右双栏布局，将原图和超分结果并排展示。原图下载按钮保留在左侧区域下方，超分结果和对比图下载按钮放置在右侧区域下方，使布局更符合用户观察习惯。

### 4.4 批量处理与 ZIP 下载功能实现

批量处理功能是本文系统的重要扩展之一。用户点击“批量处理”后可一次选择多张图片，再通过“开始批量超分”统一提交到 `/batch_process` 接口。后端遍历所有输入文件，逐张完成解码、推理、结果保存和对比图生成，并将结果信息返回前端。为支持后续批量下载，系统同时将结果缓存到 `BATCH_RESULTS_CACHE` 中，并在 `batch_outputs/<batch_id>/` 路径下保存对应文件。

在批量下载方面，系统新增 `/batch_download_zip/<batch_id>/output` 和 `/batch_download_zip/<batch_id>/compare` 两个接口。用户点击“全部下载超分结果”或“全部下载对比图”后，后端会优先从内存缓存中读取 `Base64` 数据，使用 `BytesIO` 和 `zipfile` 在内存中完成 ZIP 打包，再以流的形式返回浏览器下载。该方案避免了中文路径下的磁盘读写问题，提高了批量下载的稳定性和兼容性。

### 4.5 对比图与可视化展示实现

为便于观察超分辨率效果，系统提供了结果放大查看与对比图下载功能。后端通过 `make_compare_image()` 函数构造原图与结果图的拼接画布，在顶部添加标题，在底部标注输入输出尺寸信息。前端则通过弹窗组件实现图片放大查看，用户点击原图或结果图即可在模态框中查看大图细节。该功能提升了系统的交互体验，也便于论文中的案例展示。

### 4.6 评测模块实现

实验评测模块由 `evaluate.py` 实现。与单纯展示效果不同，该脚本支持对公开数据集进行标准化测试，其核心流程如下。

1. 读取高分辨率测试图像（HR/GT）。  
2. 对 HR 图像执行 `mod_crop`，保证尺寸可被放大倍率整除。  
3. 使用双三次下采样生成 LR 图像，并默认保存在 `datasets/<dataset>/LR_bicubic/X4/` 下。  
4. 调用指定模型对 LR 图像进行超分辨率重建，生成 SR 图像。  
5. 计算 PSNR、SSIM 和 LPIPS 指标，并保存逐图像统计结果。  
6. 导出 `model_compare.csv`、`summary.txt` 和可视化对比图，供论文分析使用。

该脚本不仅支持 `RealESRGAN_x4plus` 与 `realesr-general-x4v3`，还扩展支持 `Bicubic` 和 `ESRGAN_x4` 作为对比基线。进一步地，本文新增了 `LPIPSMetric` 类，用于在项目本地可写目录中缓存感知模型权重，从而稳定完成 LPIPS 计算。

### 4.7 核心代码说明

为了突出系统的实现过程，本文对几个核心函数进行简要说明。

1. `init_model()`：负责根据模型名称初始化网络与权重路径，并将加载完成的模型写入缓存。  
2. `process_image()`：封装单张图像超分流程，是前端单图像处理与批处理的共同调用入口。  
3. `handle_batch_process()`：后端批处理主入口，负责循环处理多图并组织返回结果。  
4. `handle_batch_zip_download()`：负责根据批次编号将超分结果或对比图打包为 ZIP 文件返回前端。  
5. `load_or_generate_lr()`：评测时自动读取或生成双三次退化图像，保证不同模型测试使用相同 LR 输入。  
6. `evaluate_one_model()`：遍历测试集、完成 SR 推理、计算指标并导出逐图像统计结果。  
7. `LPIPSMetric.calculate()`：将图像转换至感知特征空间并输出 LPIPS 值，用于补充感知质量分析。

通过上述实现，本文不仅完成了图像超分辨率系统的功能构建，也形成了较完整的评测与分析闭环。

## 第5章 系统测试与实验结果分析

### 5.1 实验目的与实验环境

本章实验主要用于验证以下三个方面的内容：  
一是系统是否能够稳定完成图像超分辨率重建与结果展示；  
二是不同模型在公开数据集上的客观指标与感知指标表现；  
三是本文扩展实现的批处理、结果导出和图表生成功能是否满足论文分析需求。

实验环境采用本地 Windows 平台，模型推理与评测均在当前项目环境下完成。实验脚本位于 `evaluate.py`，绘图脚本位于 `plot_evaluation_results.py`。

### 5.2 数据集与测试方案

#### 5.2.1 测试数据集

本文选取 Set5、Set14 和 BSD100 三个公开数据集作为测试对象。这三组数据集是超分辨率研究中较常见的标准测试集，涵盖了人物、动物、自然纹理和复杂场景等多种图像内容，能够较全面地反映模型在不同图像类型上的表现。

其中，`datasets/<dataset>/HR/` 存放高分辨率图像，即测试中的 Ground Truth（GT）。测试时以 HR 图像为基准，经双三次下采样得到对应低分辨率图像，并保存至 `LR_bicubic/X4` 路径。由于不同模型均以同一套 LR 输入作为测试起点，因此实验结果具有可比性。

#### 5.2.2 对比模型

本文选取四种模型进行比较。

1. `Bicubic`：双三次插值基线，用于反映传统插值方法性能。  
2. `ESRGAN_x4`：经典感知型超分辨率模型。  
3. `RealESRGAN_x4plus`：Real-ESRGAN 通用模型。  
4. `realesr-general-x4v3`：轻量化 Real-ESRGAN 通用模型。

#### 5.2.3 评价指标

为从不同角度衡量模型表现，本文采用 PSNR、SSIM 和 LPIPS 三项指标。  
其中，PSNR 与 SSIM 侧重重建保真度，LPIPS 更侧重视觉感知相似性。  
实验中放大倍率设为 `×4`，`crop_border` 设为 `4`，并在 Y 通道上计算 PSNR 和 SSIM。

### 5.3 定量实验结果

#### 5.3.1 Set5 数据集结果

表5-1 给出了各模型在 Set5 数据集上的测试结果。

| 模型 | PSNR/dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 27.6918 | 0.8102 | 0.3470 |
| ESRGAN_x4 | 17.2647 | 0.3670 | 0.5345 |
| RealESRGAN_x4plus | 25.4259 | 0.7851 | 0.1718 |
| realesr-general-x4v3 | 25.5326 | 0.7933 | 0.1806 |

从表中可以看出，Bicubic 在 Set5 上获得了最高的 PSNR 和 SSIM，而 `RealESRGAN_x4plus` 与 `realesr-general-x4v3` 的 LPIPS 明显更低，说明二者在视觉感知质量方面表现更优。其中，`realesr-general-x4v3` 在 PSNR 与 SSIM 上略优于 `RealESRGAN_x4plus`，表明轻量模型在该数据集上同样具有较强竞争力。

#### 5.3.2 Set14 数据集结果

表5-2 给出了各模型在 Set14 数据集上的测试结果。

| 模型 | PSNR/dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 25.0055 | 0.7054 | 0.4603 |
| ESRGAN_x4 | 15.7702 | 0.2534 | 0.6229 |
| RealESRGAN_x4plus | 23.6790 | 0.6678 | 0.2453 |
| realesr-general-x4v3 | 23.7539 | 0.6827 | 0.2564 |

在 Set14 上，Bicubic 仍然在 PSNR 和 SSIM 上占优，但两种 Real-ESRGAN 模型在 LPIPS 上大幅领先，说明其恢复结果在感知层面更接近参考图像。`RealESRGAN_x4plus` 的 LPIPS 仅为 `0.2453`，明显优于 Bicubic 的 `0.4603`，体现了感知增强模型在视觉真实感上的优势。

#### 5.3.3 BSD100 数据集结果

表5-3 给出了各模型在 BSD100 数据集上的测试结果。

| 模型 | PSNR/dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 24.6329 | 0.6648 | 0.5285 |
| ESRGAN_x4 | 15.9418 | 0.2670 | 0.6021 |
| RealESRGAN_x4plus | 23.4753 | 0.6194 | 0.2858 |
| realesr-general-x4v3 | 23.7871 | 0.6369 | 0.3083 |

在样本规模更大的 BSD100 数据集上，Bicubic 的失真指标依然较高，但两种 Real-ESRGAN 模型在 LPIPS 指标上保持明显优势，说明其在复杂自然图像上能够提供更符合视觉感知的恢复结果。`realesr-general-x4v3` 在 PSNR 与 SSIM 上略高于 `RealESRGAN_x4plus`，而 `RealESRGAN_x4plus` 在 LPIPS 上更优，说明二者在“保真度”与“感知质量”上各有侧重。

### 5.4 实验结果综合分析

#### 5.4.1 关于 Bicubic 指标较高的分析

从三组数据集结果看，Bicubic 在 PSNR 和 SSIM 上表现较强，甚至在部分数据集上高于 Real-ESRGAN 模型。这一结果并不矛盾，主要原因在于本文测试集的 LR 图像由 HR 图像经标准双三次下采样得到，而 Bicubic 正好与该退化方式高度匹配，因此在像素级误差上占据优势。相比之下，Real-ESRGAN 的训练目标是适应更加复杂的真实退化场景，模型会在一定程度上牺牲像素保真度，以换取更自然的纹理与感知效果。

#### 5.4.2 关于 LPIPS 指标的分析

LPIPS 结果表明，`RealESRGAN_x4plus` 和 `realesr-general-x4v3` 在三组数据集上均明显优于 Bicubic 和 ESRGAN_x4，说明本文采用的 Real-ESRGAN 系列模型在视觉感知质量方面具有优势。这也验证了在真实场景图像增强任务中，仅使用 PSNR 和 SSIM 评价模型是不够的，还应结合 LPIPS 等感知指标进行综合分析。

#### 5.4.3 关于两种 Real-ESRGAN 模型差异的分析

在本次测试中，`realesr-general-x4v3` 在 PSNR 和 SSIM 上略优于 `RealESRGAN_x4plus`，而 `RealESRGAN_x4plus` 在 LPIPS 上通常更低。说明轻量模型在当前双三次退化测试集上具有较好的结构恢复能力，而通用模型在感知质量方面更有优势。这种差异与模型设计目标有关，也体现出不同模型在不同评价维度上的权衡关系。

#### 5.4.4 关于 ESRGAN_x4 指标偏低的分析

ESRGAN_x4 在三组数据集上的 PSNR、SSIM 和 LPIPS 表现均不理想，说明其在当前测试协议下与数据退化形式存在明显不匹配。由于 ESRGAN 更偏向感知型纹理生成，且其训练环境与 Real-ESRGAN 不同，在盲超分辨率与真实退化适应能力方面不如后者，因此本实验中其结果主要作为历史基线参考，而不是当前系统部署首选模型。

### 5.5 图表结果说明

为便于论文插图，本文已在项目中生成三张可直接插入论文的柱状图。

1. PSNR 对比图：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\plots_4models_lpips\model_compare_psnr.png`  
2. SSIM 对比图：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\plots_4models_lpips\model_compare_ssim.png`  
3. LPIPS 对比图：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\plots_4models_lpips\model_compare_lpips.png`

论文排版时可分别作为“图5-1 PSNR 对比图”“图5-2 SSIM 对比图”“图5-3 LPIPS 对比图”插入，并在图下对指标变化趋势进行解释。

### 5.6 系统功能测试

除定量指标实验外，本文还对系统功能进行了验证，结果如下。

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

系统测试表明，当前实现已较完整地覆盖毕业设计任务书中提出的主要功能需求。

### 5.7 本章小结

本章从数据集、测试流程、定量结果、图表输出和系统功能测试五个方面对本文系统进行了验证。实验结果表明，本文构建的系统能够稳定完成图像超分辨率处理与多模型评测任务；Real-ESRGAN 系列模型在 LPIPS 指标上明显优于传统插值方法，说明其在视觉感知质量方面更具优势；同时，系统的前后端交互、批处理和结果导出功能均能稳定运行，达到了毕业设计系统原型实现与实验验证的目标。

## 第6章 总结与展望

### 6.1 工作总结

本文围绕“基于 Real-ESRGAN 的图像超分辨率系统设计与实现”这一课题，完成了图像超分辨率相关理论梳理、Real-ESRGAN 模型分析、系统需求分析、前后端功能实现以及实验评测扩展等工作。具体而言，本文基于官方公开预训练模型构建了图像超分辨率后端推理模块，基于 Web 技术实现了可视化交互界面，并在系统中加入了单图像处理、批量处理、对比展示、细节放大、结果下载和 ZIP 批量导出等功能。

在实验部分，本文实现了标准化评测脚本，构建了从 HR 图像自动生成 LR 图像、进行多模型推理、计算 PSNR/SSIM/LPIPS 指标到输出统计图表的完整流程。在 Set5、Set14 和 BSD100 三个公开数据集上的实验表明，本文系统能够稳定工作，两种 Real-ESRGAN 模型在 LPIPS 等感知指标上具有明显优势，验证了系统在图像增强和画质修复场景中的应用潜力。

### 6.2 主要成果与特点

本文的主要成果可以概括为以下几点。

1. 完成了基于 Real-ESRGAN 的图像超分辨率系统设计与实现，实现了从模型调用到界面交互的完整闭环。  
2. 在原有基础上完善了批处理功能，并实现了批量结果和批量对比图的 ZIP 一键下载。  
3. 扩展实现了评测模块，支持 Bicubic、ESRGAN 和 Real-ESRGAN 系列模型对比。  
4. 新增 LPIPS 指标计算与绘图功能，使论文实验分析从“保真度评价”扩展到“感知质量评价”。  
5. 形成了一套与当前项目代码一致的论文实验流程，可直接支持毕业论文撰写与答辩展示。

### 6.3 不足与展望

尽管本文完成了系统设计、实现与实验验证，但仍存在一些不足。

第一，本文主要基于官方预训练权重开展系统集成与应用验证，未结合特定数据集进行再训练或微调，因此在标准双三次退化测试集上的 PSNR 仍有进一步提升空间。  
第二，当前系统主要面向本地单机部署，尚未进一步扩展为多用户并发访问的生产级服务。  
第三，实验评价主要集中在 Set5、Set14 和 BSD100，后续仍可加入 Urban100、Manga109 以及更多真实场景数据集。  
第四，当前系统尚未加入用户登录、结果历史记录、任务队列管理等更完整的应用功能。

后续研究可从以下几个方向展开：一是针对双三次退化或特定场景数据进行微调训练，以提高 PSNR 和 SSIM；二是引入更多新型模型，如扩散式真实场景超分模型，进一步改善感知质量；三是优化系统部署方式，提升并发能力和运行效率；四是加入更多评价维度，如运行时间、显存占用和主观用户打分，以形成更加全面的系统评估体系。

## 参考文献

[1] Dong C, Loy C C, He K, et al. Learning a Deep Convolutional Network for Image Super-Resolution[C]//European Conference on Computer Vision. Cham: Springer, 2014: 184-199.  
[2] Kim J, Lee J K, Lee K M. Accurate Image Super-Resolution Using Very Deep Convolutional Networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 1646-1654.  
[3] Lim B, Son S, Kim H, et al. Enhanced Deep Residual Networks for Single Image Super-Resolution[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2017: 1132-1140.  
[4] Ledig C, Theis L, Huszár F, et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 4681-4690.  
[5] Zhang Y, Li K, Li K, et al. Image Super-Resolution Using Very Deep Residual Channel Attention Networks[C]//Proceedings of the European Conference on Computer Vision. 2018: 286-301.  
[6] Wang X, Yu K, Wu S, et al. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks[C]//Proceedings of the European Conference on Computer Vision Workshops. 2018.  
[7] Zhang R, Isola P, Efros A A, et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 586-595.  
[8] Timofte R, De Smet V, Van Gool L. A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution[C]//Asian Conference on Computer Vision. Cham: Springer, 2014: 111-126.  
[9] Martin D, Fowlkes C, Tal D, et al. A Database of Human Segmented Natural Images and Its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics[C]//Proceedings of the IEEE International Conference on Computer Vision. 2001: 416-423.  
[10] Liang J, Cao J, Sun G, et al. SwinIR: Image Restoration Using Swin Transformer[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. 2021: 1833-1844.  
[11] Zhang K, Liang J, Van Gool L, et al. Designing a Practical Degradation Model for Deep Blind Image Super-Resolution[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 4791-4800.  
[12] Wang X, Xie L, Dong C, et al. Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. 2021: 1905-1914.  
[13] Bhat G, Gharbi M, Chen J, et al. Self-Supervised Burst Super-Resolution[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 10605-10614.  
[14] Park J, Son S, Lee K M. Content-Aware Local GAN for Photo-Realistic Super-Resolution[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 10585-10594.  
[15] Yin Z, Liu M, Li X, et al. MetaF2N: Blind Image Super-Resolution by Learning Efficient Model Adaptation from Faces[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 13033-13044.  
[16] Wei P, Sun Y, Guo X, et al. Towards Real-World Burst Image Super-Resolution: Benchmark and Method[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 13233-13242.  
[17] Ray A, Kumar G, Kolekar M H. CFAT: Unleashing Triangular Windows for Image Super-Resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 26120-26129.  
[18] Wu R, Yang T, Sun L, et al. SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 25456-25467.  
[19] Korkmaz C, Tekalp A M, Dogan Z. Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 5926-5936.  
[20] Zhang L, Li Y, Zhou X, et al. Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 2856-2865.  
[21] Mei K, Talebi H, Ardakani M, et al. The Power of Context: How Multimodality Improves Image Super-Resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025: 23141-23152.  
[22] Chen B, Li G, Wu R, et al. Adversarial Diffusion Compression for Real-World Image Super-Resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025: 28208-28220.  
[23] Sun L, Wu R, Ma Z, et al. Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025: 2333-2343.  
[24] 唐艳秋, 潘泓, 朱亚平, 等. 图像超分辨率重建研究综述[J]. 电子学报, 2020, 48(7): 1407-1420.  
[25] 王睿琪. 图像超分辨率重建综述[J]. 计算机科学与应用, 2024, 14(2): 350-359.  

## 致谢

在本次毕业设计与论文撰写过程中，我系统梳理了图像超分辨率相关理论，并结合项目代码完成了 Real-ESRGAN 图像超分辨率系统的设计、实现与测试。在此过程中，我得到了指导教师在课题方向、系统实现和论文写作方面的帮助与指导，使我能够逐步完成从算法理解、功能实现到实验分析的全过程。

同时，感谢开源社区提供的 Real-ESRGAN 项目基础代码和相关公开模型权重，为本课题的实现和验证提供了良好的技术基础。感谢公开数据集和相关论文作者，为本课题的实验设计与结果分析提供了参考依据。通过本次毕业设计，我不仅加深了对图像超分辨率与深度学习方法的理解，也提高了自身在系统集成、实验评测和技术写作方面的综合能力。

## 附录A 论文插图与结果文件说明

1. PSNR 图：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\plots_4models_lpips\model_compare_psnr.png`  
2. SSIM 图：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\plots_4models_lpips\model_compare_ssim.png`  
3. LPIPS 图：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\plots_4models_lpips\model_compare_lpips.png`  
4. Set5 实验结果：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\Set5_4models_lpips\model_compare.csv`  
5. Set14 实验结果：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\Set14_4models_lpips\model_compare.csv`  
6. BSD100 实验结果：`C:\Users\Lenovo\Desktop\毕设项目\Real-ESRGAN\evaluation_results\Bsd100_4models_lpips\model_compare.csv`

## 附录B 测试命令说明

```powershell
python .\evaluate.py --hr-dir .\datasets\Set5\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set5_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

python .\evaluate.py --hr-dir .\datasets\Set14\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set14_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

python .\evaluate.py --hr-dir .\datasets\Bsd100\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Bsd100_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10
```
](https://chat.deepseek.com/a/chat/s/e9d8ef86-cd94-489e-849d-f1d73aa1609a)

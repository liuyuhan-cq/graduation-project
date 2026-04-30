# 基于Real-ESRGAN的图像超分辨率系统设计与实现

> 排版说明  
> 1. 首页、诚信承诺书、版权声明等前置页面按学校模板单独填写。  
> 2. 正文中的英文、数字、变量名、文件名、模型名统一采用 `Times New Roman`。  
> 3. 文中公式建议在 Word 中使用“公式编辑器”重新录入。  
> 4. 目录、图目录、表目录建议使用 Word 自动生成。  
> 5. 本稿已根据学校模板要求调整章节标题、表格采用三线表格式，图表前后留空行，参考文献格式已补全。  
> 6. 实验环境信息已补充，表格内指标单位已标注。

## 摘要

图像超分辨率旨在从低分辨率图像中恢复高分辨率细节，是计算机视觉与图像处理领域的重要研究课题，广泛应用于老旧照片修复、监控视频增强、移动端成像优化等场景。传统插值方法计算开销小，但在复杂降质条件下难以恢复细节，常导致模糊、锯齿和边缘失真。生成对抗网络的进展推动了感知驱动型超分辨率模型的发展，Real-ESRGAN通过高阶退化建模和改进判别器设计，显著提升了对真实场景低质图像的适应能力。

本研究设计并实现了一个基于 Real-ESRGAN 的图像超分辨率系统。系统采用 Python、PyTorch 和 OpenCV 实现后端推理服务，前端提供单图像超分、批量处理、对比展示、细节放大和一键 ZIP 下载等功能。此外，研究扩展实现了多模型评测模块，支持 Bicubic、ESRGAN、RealESRGAN_x4plus 和 realesr-general-x4v3 四种模型对比，并引入 PSNR、SSIM 和 LPIPS 指标。

在 Set5、Set14 和 BSD100 三个公开数据集上的实验结果表明，系统稳定完成超分辨率任务。以 RealESRGAN_x4plus 为例，其在三个数据集上的平均 LPIPS 分别为 0.1718、0.2453 和 0.2858，显著优于 Bicubic（LPIPS 分别为 0.3470、0.4603 和 0.5285）和 ESRGAN_x4。realesr-general-x4v3 在 PSNR 与 SSIM 上略优于 x4plus，表明两者在保真度与感知质量之间的不同侧重。实验结果验证了本系统在真实场景图像增强中的可用性和工程应用价值。

关键词：图像超分辨率；Real-ESRGAN；生成对抗网络；Web系统；LPIPS

## Abstract

Image super-resolution aims to recover high-resolution details from low-resolution images and has great potential in old photo restoration, surveillance enhancement, mobile imaging, and digital content repair. Traditional interpolation methods are easy to deploy but often fail to reconstruct textures under complex degradations. Generative adversarial networks have driven substantial progress in perceptual-driven super-resolution, among which Real-ESRGAN achieves strong real-world adaptability via high-order degradation modeling and an improved discriminator.

This thesis designs and implements a Real-ESRGAN-based image super-resolution system. With a backend inference service at its core, the system uses Python, PyTorch and OpenCV for model loading, preprocessing and reconstruction, and provides a Web interface for single-image enhancement, batch processing, synchronized input-output comparison, detail zooming, and one-click ZIP download. An evaluation module supporting Bicubic, ESRGAN, RealESRGAN_x4plus and realesr-general-x4v3 is further developed, and three metrics—PSNR, SSIM and LPIPS—are adopted.

Experimental results on Set5, Set14, and BSD100 show that the system handles super-resolution tasks stably. For RealESRGAN_x4plus, the average LPIPS values on the three datasets are 0.1718, 0.2453 and 0.2858, respectively—substantially better than those of Bicubic (0.3470, 0.4603, 0.5285) and ESRGAN_x4. The realesr-general-x4v3 variant achieves slightly higher PSNR and SSIM, while x4plus achieves lower LPIPS, revealing a balanced trade-off between fidelity and perceptual quality. These results confirm the system's practicality and engineering value for real-world image enhancement.

Keywords: image super-resolution; Real-ESRGAN; generative adversarial network; Web system; LPIPS

## 第1章 引言

### 1.1 研究背景与意义

#### 1.1.1 现实意义

数字图像已广泛应用于安防监控、智能终端、医学辅助诊断、数字档案修复和网络媒体传播。受限于采集设备性能、传输压缩、拍摄抖动和噪声干扰，海量图像存在分辨率不足、纹理细节缺失与边缘模糊等问题。单纯依赖硬件升级代价高昂，且对历史图像、老旧照片与网络压缩图像无能为力。因此，利用软件算法提升低分辨率图像质量，在安防取证、医学影像增强、历史档案数字化等领域具有显著的现实需求和应用价值。

图像超分辨率（Super-Resolution, SR）技术能够在不改变原始采集硬件的前提下提升图像清晰度，为后续检测、识别、理解与展示提供更丰富的细节。实际场景中，该技术可用于修复旧照片、增强监控截图、提升移动端显示效果、辅助文本图像识别，并为画质增强和图像修复提供前置支持。特别是在真实降质复杂、图像来源多样的应用环境下，构建一套兼顾重建效果与交互易用性的超分辨率系统，具有重要的工程意义。

#### 1.1.2 研究价值

从数学上看，图像超分辨率是一个典型的病态逆问题：一幅给定的低分辨率图像，理论上存在无穷多种高分辨率重建结果与之对应——较小尺寸的观测约束远不足以唯一确定高维的原始信号。这一固有不确定性使得传统方法（如基于插值或稀疏编码的算法）在简单退化假设下尚可复原图像轮廓，但面对未知的真实世界退化时，往往无法合理填补缺失的高频纹理[1-4]。

近年来，以生成对抗网络（Generative Adversarial Network, GAN）为代表的深度学习方法为上述困境提供了新的解决思路。有别于仅追求逐像素精度的传统模型，GAN通过对抗训练迫使生成器不仅“画得像”，更要“画得真”——即生成在纹理细节上符合自然图像统计分布的高频信息[2]。这一特性使其天然适合处理感知质量驱动的超分辨率任务，并将该领域的研究重心从“单纯的像素保真度”逐步迁移至“真实场景、感知质量与工程落地”的综合方向。

Real-ESRGAN 是面向真实场景盲超分辨率的代表性模型。所谓“盲超分辨率”，是指在低分辨率图像的降质过程（模糊核、噪声、压缩方式等）完全未知的前提下进行复原，这远比假设标准双三次下采样的传统设定更具挑战性，也更贴近现实应用。Real-ESRGAN 在继承 ESRGAN 感知增强能力的基础上，引入高阶降质建模和谱归一化 U-Net 判别器，在真实图像恢复任务中表现出较强的适应性[5-6]。因此，围绕 Real-ESRGAN 开展系统设计与应用研究，不仅有助于理解真实场景超分辨率算法的关键机理，也能推动该类模型向实用化系统转化。

### 1.2 国内外研究现状

#### 1.2.1 国外研究现状

国外对图像超分辨率的研究始于插值重建、稀疏表示和样本学习等方向。传统插值方法在恢复高频纹理方面存在较大局限，而基于样本字典或稀疏表示的方法虽有所改善，但计算复杂且依赖先验。Dong 等首次将卷积神经网络应用于单幅图像超分辨率，为深度学习超分辨率研究奠定了基础[1]。此后，VDSR、EDSR 和 RCAN 等模型通过加深网络层数、引入残差连接与通道注意力机制，不断刷新像素级重建性能。

随着对 PSNR 和 SSIM 的追求逐渐饱和，研究者认识到高客观指标并不总能带来更优的视觉观感。Ledig 等提出 SRGAN，将 GAN 引入超分辨率任务并在纹理重建上取得重要突破[2]；Wang 等进一步提出 ESRGAN，通过改进残差块结构和感知损失设计，显著增强了纹理表现力[3]。此后，感知质量与失真指标之间的权衡成为该领域的重要研究主题。

真实场景图像恢复需求的增加促使研究重点向盲超分辨率和真实降质建模转移。例如，在实际监控、手机拍摄等场景中，图像常经历不同程度的模糊、噪声和压缩伪影叠加，而多数高 PSNR 模型在这些场景下会出现明显的纹理缺失或结构失真。Wang 等提出 Real-ESRGAN，通过高阶降质建模和谱归一化 U-Net 判别器取得真实图像恢复中的领先视觉表现[5-6]。近年来，SwinIR 等模型将 Transformer 引入超分辨率任务，增强了全局上下文建模能力；SeeSR 等方法则探索语义先验、扩散模型与多模态信息在真实场景超分中的应用。在架构方面，HAT等模型通过激活更多像素参与超分重建[16]；在语义引导方面，SeeSR等探索语义先验对真实场景超分的增强效果[17]。国外研究已从“提高像素精度”扩展到“兼顾感知质量、泛化能力与推理效率”的综合目标。

#### 1.2.2 国内研究现状

国内图像超分辨率研究近年进展迅速。唐艳秋等系统梳理了传统方法与深度学习超分的发展脉络[11]；王睿琪从单图像和多图像两个角度总结了超分辨率的发展现状与挑战[12]。在算法层面，国内团队不仅关注模型深度与结构优化，也逐步重视复杂降质建模、轻量化部署和实际系统实现。越来越多的研究以开源预训练模型为基础，开展应用适配、模块集成和评测扩展工作，以提升工程可用性。国内近年也涌现出多篇针对超分辨率技术的中文综述[13-15][21]，从不同角度系统梳理了该领域的研究进展。

### 1.3 现有研究存在的主要问题

综合国内外研究现状，当前图像超分辨率研究仍存在以下主要问题。

第一，真实场景降质复杂，模型泛化能力不足。多数高指标模型假设低分辨率图像由标准双三次下采样产生，而真实场景中模糊、噪声、压缩失真和传感器噪声常叠加共存[14]。研究表明，在此类混合退化条件下，仅训练于双三次退化的模型会出现 PSNR 显著下降及细节伪造等现象，导致在真实图像上出现过度平滑或结构失真[7-8]。

第二，客观指标与主观感知之间的矛盾突出。以 PSNR 和 SSIM 为代表的失真指标更侧重像素级一致性，而以 GAN 或扩散模型为代表的感知增强方法更强调纹理真实感，二者常常处于权衡之中[7-8]。

第三，算法与系统脱节现象明显。大量研究聚焦于模型本身，忽略了文件上传、交互预览、批量处理、结果管理与一键下载等工程化功能，导致算法难以直接服务于实际用户。

第四，评测流程不够完整。部分系统仅展示视觉结果，缺少标准化测试集、统一降质处理以及多维度指标评估机制，不利于性能的客观比较与复现验证。

### 1.4 主要内容和工作安排

本文主要内容是基于 Real-ESRGAN 的图像超分辨率系统的设计与实现。首先，对图像超分辨率及 Real-ESRGAN 的关键理论进行了梳理，重点分析了生成器结构、判别器设计、高阶降质建模策略和损失函数设计。接着，基于 Python、PyTorch 和 OpenCV 封装了 Real-ESRGAN 推理流程，构建了面向实际应用的图像超分辨率后端服务，并采用 Web 技术实现了系统的前端交互界面。随后，扩展实现了多模型评测脚本，支持 Bicubic、ESRGAN、RealESRGAN_x4plus 和 realesr-general-x4v3 四种模型的对比，并引入 PSNR、SSIM 和 LPIPS 三项评价指标。最后，在 Set5、Set14 和 BSD100 三个公开数据集上完成了系统测试与实验分析，从保真度与感知质量两个维度讨论了不同模型的表现差异。需要说明的是，本文以“系统设计与实现”及“模型应用验证”为研究重点，受硬件资源与毕业设计周期限制，未对 Real-ESRGAN 进行从头训练，而是基于官方公开预训练权重完成模型集成、接口封装、评测扩展与可视化系统实现。

本文一共由六章组成，按照以下架构进行编写。

第1章为引言，介绍了图像超分辨率的研究背景与意义。分析了当前真实场景图像降质复杂、传统方法泛化能力不足的问题，并结合国内外研究现状，阐述了 Real-ESRGAN 作为盲超分辨率代表性模型的研究价值与应用前景。

第2章概述了图像超分辨率与 Real-ESRGAN 的相关理论基础，包括超分辨率问题的数学描述、RRDB 生成器结构、高阶降质建模策略、带谱归一化的 U-Net 判别器设计以及多损失联合优化机制。同时，详细介绍了 PSNR、SSIM 和 LPIPS 三类图像质量评价指标，以及感知-失真权衡理论对模型选择的指导意义。

第3章对系统进行了需求分析。首先从技术、经济和操作三个角度分析了系统的可行性，然后从功能需求和非功能需求两个维度明确了系统应具备的能力，为后续设计工作奠定了基础。

第4章给出了系统的详细设计方案。首先描述了系统的总体架构，将系统划分为前端交互、后端推理、批处理管理和实验评测四个模块；然后分别对单图像超分、批量处理、对比图与可视化、实验评测四个功能模块的设计思路进行了说明；最后给出了系统的业务流程设计。

第5章介绍了系统的功能实现与测试情况。依次阐述了开发环境、各功能模块的具体实现细节与界面展示、系统功能测试结果，以及基于三个公开数据集和四项评价指标的算法测试实验设计与结果分析。

第6章对本文的研究工作进行了总结，归纳了主要成果，分析了当前工作的不足之处，并对未来的改进方向进行了展望。

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

该问题的核心困难在于其具有典型的不适定性。从信息论角度看，低分辨率图像的信息量远小于高分辨率图像，降质过程本质上是信息丢失，而逆映射是一对多的：同一个低分辨率观测可能对应无穷多种高分辨率重建结果。这意味着即使 $I_{LR}$ 被完美重建为某种高分辨率形式，也无法保证该结果与真实的 $I_{HR}$ 一致。因此，如何从有限的观测中恢复既满足数据一致性又具有感知真实性的细节，成为超分辨率研究的根本挑战[1-3]。

### 2.2 Real-ESRGAN 模型原理

#### 2.2.1 生成器结构

Real-ESRGAN 的生成器沿用了 ESRGAN 中提出的 RRDB（Residual-in-Residual Dense Block）结构。该结构的核心设计思想体现在三个层面：其一，残差中嵌套残差，使网络可以在不同粒度上学习恒等映射偏差，而非直接预测全部像素；其二，密集连接使每一层都能直接接收前面所有层的特征，增强了特征的复用效率和梯度流动；其三，移除了传统残差块中常见的批归一化（Batch Normalization）层——ESRGAN 原论文的消融实验表明，BN 层在深层 GAN 训练中会产生伪影，移除后可获得更干净的高频纹理[3,5]。这三项设计共同提高了网络的表达能力和训练稳定性，使得 RRDB 在纹理细节恢复方面较一般深层卷积网络更具优势。

在本文所集成的模型中，`RealESRGAN_x4plus` 采用 23 个 RRDB 模块，面向通用场景追求最高的重建质量；`RealESRGAN_x4plus_anime_6B` 采用 6 个 RRDB 模块，在保持对动漫图像适配能力的同时减小了模型规模；`realesr-general-x4v3` 则采用更轻量的 `SRVGGNetCompact` 结构，以进一步降低推理延迟和内存占用。在实际部署中，可根据应用场景（高画质/动漫优化/实时处理）灵活选择对应的模型版本。Real-ESRGAN 生成器的整体结构如图2-1所示。

![图2-1 Real-ESRGAN生成器结构简图](images/fig2-1.png)

图2-1 Real-ESRGAN生成器结构简图

#### 2.2.2 高阶降质建模

Real-ESRGAN 的重要改进之一在于提出了更贴近真实场景的高阶降质建模策略[5-6]。不同于传统方法通常仅考虑单次模糊与一次双三次下采样，该方法通过多轮随机模糊、缩放、噪声扰动和 JPEG 压缩的叠加，模拟真实图像在多次采集、传输和再压缩等过程中可能经历的复杂退化。这一策略的核心价值在于：通过在训练阶段极大扩展降质分布的覆盖范围，缩小了合成训练数据与真实测试输入之间的分布差异，从而在不依赖任何真实世界配对训练数据的前提下，使模型获得了对未知真实退化的泛化能力。Real-ESRGAN 原论文在多个真实场景测试集上的实验已验证了该策略的有效性[5]。需要指出的是，该高阶降质建模策略已在预训练模型训练阶段完成，系统推理时无需显式实现降质过程，而是直接调用预训练好的模型对输入图像进行超分辨率处理。

#### 2.2.3 判别器结构

在判别器设计方面，Real-ESRGAN 相较于 ESRGAN 常用的 PatchGAN 判别器，进一步引入了带谱归一化的 U-Net 判别器[5]。选择 U-Net 结构的主要原因在于：PatchGAN 判别器仅输出整体图像的单一真假判断，无法为生成器提供像素级的局部反馈；而 U-Net 判别器在每个空间位置都产生判别信号，能够对图像的局部纹理进行精细监督，有助于生成器在细节处产生更真实的纹理。

谱归一化方面，其原理是通过约束判别器中每一层的谱范数，将判别器的 Lipschitz 常数限制在一定范围内，从而防止判别器参数在对抗训练中过度增长，有效缓解训练过程中的梯度爆炸和振荡问题[5]。这一设计使得 Real-ESRGAN 在更大的退化空间上仍能稳定训练，并最终输出结构一致性和纹理真实性更优的重建结果。判别器的 U-Net 结构如图2-2所示。

![图2-2 Real-ESRGAN判别器结构简图](images/fig2-2.png)

图2-2 Real-ESRGAN判别器结构简图

#### 2.2.4 损失函数设计

Real-ESRGAN 通过内容损失、感知损失与对抗损失的联合优化，在重建精度与视觉感知质量之间取得平衡。其中，感知损失的思想可追溯至 Johnson 等人提出的工作[4]：将生成图像与目标图像分别输入预训练的 VGG-16 网络，比较它们在高层特征空间中的距离，从而在语义层面而非像素层面对生成结果进行评价。Real-ESRGAN 在此基础上继承并改进，其总损失可表示为：

$$
\mathcal{L}_{total}=\lambda_{1}\mathcal{L}_{1}+\lambda_{p}\mathcal{L}_{percep}+\lambda_{g}\mathcal{L}_{GAN}
$$

其中 $\mathcal{L}_{1}$ 为像素空间的 L1 损失，约束重建结果的整体纹理结构与目标图像在像素层面尽量一致；$\mathcal{L}_{percep}$ 为感知损失，在 VGG 网络的特征空间中比对生成结果与目标图像，强调高层语义结构的一致性；$\mathcal{L}_{GAN}$ 为对抗损失，驱使生成器产生更符合自然图像分布的细节纹理。三类损失的协同优化，使模型既能够保持整体结构准确，又能够在高频细节层面获得较好的真实感。

### 2.3 图像质量评价指标

#### 2.3.1 PSNR

峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）是衡量重建图像与参考图像之间像素级误差的常用指标，其基础为均方误差（MSE）：

$$
MSE=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\left(I_{HR}(i,j)-\hat{I}_{HR}(i,j)\right)^2
$$

$$
PSNR=10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

其中 $MAX_I$ 表示像素最大值，在 8 位图像中通常取 255。PSNR 越高，说明重建图像在像素意义上越接近参考图像。然而，PSNR 主要反映像素误差，并不能充分体现人眼对视觉质量的主观感受，因此通常需要结合其他指标进行综合评价[9]。

#### 2.3.2 SSIM

结构相似性（Structural Similarity, SSIM）从亮度、对比度和结构三个方面衡量两幅图像的相似程度，相较于 PSNR 更符合人眼视觉感知机制[10]。其表达式为：

$$
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

其中 $\mu_x,\mu_y$ 为图像均值，$\sigma_x^2,\sigma_y^2$ 为方差，$\sigma_{xy}$ 为协方差，$C_1,C_2$ 为稳定常数。SSIM 取值越接近 1，说明图像在结构层面越相似。

#### 2.3.3 LPIPS

学习感知图像块相似度（Learned Perceptual Image Patch Similarity, LPIPS）通过预训练深度网络的特征空间距离衡量两幅图像之间的感知差异[8]。LPIPS 值越低，表示两幅图像在感知层面越接近。与 PSNR 和 SSIM 相比，LPIPS 更能反映生成模型在纹理真实感、边缘锐度以及整体视觉观感方面的优劣，因此特别适用于 GAN 类感知增强模型的评价。

#### 2.3.4 感知—失真权衡

Blau 和 Michaeli 从理论上指出，图像复原任务中的失真指标与感知质量之间存在根本性权衡[7]。具体而言，随着均方误差（MSE）等失真指标的逐步降低，算法倾向于输出所有可能高分辨率解的平均值，而非从自然图像分布中采样——这使得结果虽然在数值上更接近参考图像，但在视觉上却更平滑、缺乏真实纹理。这一理论为系统设计中的模型选择提供了明确的指导原则：若目标为最大化视觉感知质量，则不应仅以 PSNR 或 SSIM 为优化目标。在本文实验中，Bicubic 插值虽然在 PSNR 上领先，但其 LPIPS 显著落后于 Real-ESRGAN 系列模型，正是这一权衡关系的直接实证。

### 2.4 Web 系统相关技术

本系统采用轻量级前后端交互架构。前端使用 HTML、CSS 和 JavaScript 实现文件选择、模型切换、结果展示、放大预览和批量下载等功能；后端基于 Python 标准库中的 `http.server` 与 `socketserver` 搭建简易 HTTP 服务，结合 PyTorch 完成模型推理，并借助 OpenCV 实现图像编解码与对比图拼接。该技术方案依赖简单、部署方便，适合毕业设计原型系统的实现需求。

## 第3章 系统需求分析

### 3.1 可行性分析

#### 3.1.1 技术可行性

系统后端采用 Python 语言与 PyTorch 深度学习框架，前端采用原生 HTML、CSS 和 JavaScript 构建，均为成熟且社区活跃的开源技术。Real-ESRGAN 模型提供了公开的预训练权重和标准推理接口，无需从零训练即可完成图像超分辨率任务。系统整体依赖较少，对运行环境要求不高，在个人计算机上即可完成开发、测试与部署，技术实现路径清晰可行。

#### 3.1.2 经济可行性

本系统所依赖的 Python 环境、PyTorch 框架、OpenCV 库以及 Real-ESRGAN 预训练模型均为开源免费资源，开发工具和运行平台均为学校或个人已具备的基础条件，无需额外采购商业软件或服务。因此，系统的开发与运行不产生额外的经济成本。

#### 3.1.3 操作可行性

系统通过 Web 浏览器提供统一的操作界面，用户无需安装任何客户端软件或深度学习环境，仅需打开网页即可完成图像上传、模型选择、超分处理和结果下载等操作。界面采用左右双栏布局，操作流程简洁直观，非专业用户也能快速上手使用。

### 3.2 功能需求分析

系统在功能层面需支持以下核心能力：(1) 用户上传单张图像并完成超分辨率重建；(2) 多种模型切换，包括通用模型、动漫模型和轻量模型；(3) 原图与超分结果以左右双栏形式同步展示，并标注输入与输出分辨率；(4) 结果图像放大查看，便于观察局部细节；(5) 批量图像处理，降低重复操作成本；(6) 单张结果下载，以及批量结果和批量对比图以 ZIP 格式一键打包下载；(7) 标准化评测，输出可直接用于论文整理的指标表格与对比图表。

### 3.3 非功能需求

在非功能层面，系统应满足以下要求：易用性方面，界面布局直观，操作流程简洁；可扩展性方面，预留接入新模型和新评价指标的接口；可维护性方面，前端、后端和评测模块之间保持相对解耦；稳定性方面，在批量处理、文件下载和模型缓存等操作中避免重复加载或数据丢失；实用性方面，既能够直观展示算法效果，也能够服务于论文实验分析。

### 3.4 本章小结

本章首先从技术、经济和操作三个方面分析了系统的可行性，然后从功能需求和非功能需求两个维度明确了系统应具备的能力。通过需求分析，确定了系统需要支持单图像超分与批量处理两种核心业务模式，并需具备模型切换、结果对比展示、ZIP 打包下载和标准化评测等完整功能链路。这些需求为后续章节的系统设计提供了明确的目标和约束。

## 第4章 系统设计

### 4.1 系统总体设计

系统在整体架构上划分为前端交互模块、后端推理模块、批处理管理模块和实验评测模块四个部分。前端交互模块负责图像上传、模型选择、结果展示、对比图下载与弹窗预览；后端推理模块负责模型初始化、权重加载、超分推理与结果编码；批处理管理模块负责多文件输入、结果缓存、文件保存与 ZIP 打包下载；实验评测模块负责测试集读取、低分辨率图像生成、多模型对比、指标计算与图表输出。

系统整体工作流程如下：用户在前端上传图像并选择模型，前端以 multipart/form-data 方式将图像发送至 /process 接口；后端完成请求解析与模型推理后，返回超分结果、拼接对比图以及尺寸信息；前端据此完成可视化展示与下载操作。对于批量处理任务，系统依次处理多张输入图像，并将结果及对比图缓存在本地目录与内存中，以便后续打包下载。

### 4.2 功能模块设计

基于第3章的需求分析，系统在功能层面设计了以下核心模块。

#### 4.2.1 单图像超分功能设计

该模块为用户提供从图像上传到结果下载的完整处理链路。用户通过前端选择图片文件并指定推理模型，前端将图像以 Base64 编码后发送至后端；后端完成解码、推理和结果编码后，将超分结果图、拼接对比图、输入输出尺寸及模型名称返回前端，前端以左右双栏布局展示原图与超分结果，并提供单张结果和对比图的下载功能。系统相关的前端页面文件（index.html、script.js）和后端服务文件（web_server.py）均围绕该模块的核心流程进行组织。

#### 4.2.2 批量处理功能设计

该模块在单图像超分模块的基础上扩展了批量处理能力。用户可一次性选择多张图片并统一提交至批处理接口，后端逐张完成推理后将结果缓存在内存与本地目录中。批量处理结果支持两种导出方式：单张逐一下载，或以 ZIP 压缩包形式一键打包全部超分结果或全部对比图。ZIP 打包采用内存动态构建方案，减少了磁盘读写开销并规避了中文路径兼容性问题。批量处理过程中生成的结果文件与缓存数据统一存放于 batch_outputs 目录下，便于管理。

#### 4.2.3 对比图与可视化功能设计

该模块负责生成原图与超分结果的拼接对比图，并为用户提供结果放大查看功能。后端通过统一定制拼接画布，在顶部添加标题、在底部标注分辨率信息。前端利用模态框组件实现大图弹窗预览，用户可点击原图或结果图查看局部细节，无需进行二次插值处理。

#### 4.2.4 实验评测功能设计

该模块为独立的离线评测工具，支持对多个超分辨率模型进行标准化定量对比。评测流程包括：读取高分辨率测试图像，通过 mod_crop 保证尺寸可被放大倍率整除，以双三次下采样生成低分辨率输入并存放于 LR_bicubic/X4 目录下，调用指定模型完成推理，计算 PSNR、SSIM 和 LPIPS 三项指标，并导出逐图像统计结果与可视化图表。评测模块支持 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3 四种模型，低分辨率图像可复用已有生成结果，也支持通过 --regenerate-lr 参数强制重新生成。评测脚本 evaluate.py 与模型权重目录 weights/ 相互配合，所有评测输出统一保存至 evaluation_results/ 目录。

### 4.3 系统业务流程设计

单图像超分业务流程为：用户通过前端界面上传图像文件，前端完成原图预览后，用户选择所需的推理模型，将请求提交至后端；后端调用 Real-ESRGAN 模型完成超分辨率推理，生成超分结果图和拼接对比图；前端接收返回的结果图像和尺寸信息后，在界面上展示原图与超分结果的对比，用户可根据需要下载单张结果或对比图。

批量处理业务流程为：用户在前端一次性选择多张图像文件，确认模型选择后统一提交批处理请求；后端遍历所有输入文件，逐张完成图像解码、模型推理、结果保存和对比图生成；处理完成后，前端展示全部图像的处理状态和结果列表；用户可选择逐张下载结果，也可通过 ZIP 一键下载功能打包获取全部超分结果或全部对比图。

评测业务流程为：读取指定数据集目录下的高分辨率 Ground Truth 图像，检查或生成对应的低分辨率输入；依次调用各模型对低分辨率图像进行超分辨率重建，保存生成的 SR 图像和对比图；计算每张图像的 PSNR、SSIM 和 LPIPS 指标，汇总为逐图像统计表；最后导出模型对比 CSV 文件、汇总文本和柱状图，供论文实验分析使用。

### 4.4 本章小结

本章从系统总体架构出发，将系统划分为四个核心模块，明确了各模块的职责和数据流转关系。随后，对单图像超分、批量处理、对比图与可视化、实验评测四个功能模块的设计思路和关键技术方案进行了说明，并给出了各模块对应的文件组织方式。最后，以分段形式描述了单图像超分、批量处理和实验评测三类核心业务流程。通过本章的设计工作，系统在需求分析文档和实际代码实现之间建立了清晰的映射关系，为后续的系统实现与测试提供了完整的蓝图。

## 第5章 系统功能实现与测试

### 5.1 开发环境

系统开发与测试在本地 Windows 平台上完成。硬件环境为 AMD Ryzen 7 5800H with Radeon Graphics 处理器，未配备独立 GPU；操作系统为 Windows 11。软件方面，开发语言为 Python 3.10，深度学习框架采用 PyTorch 1.13.1（CPU 版本），图像处理依赖 OpenCV，前端使用原生 HTML、CSS 和 JavaScript，后端基于 Python 标准库中的 `http.server` 与 `socketserver` 构建 HTTP 服务。模型推理通过 `RealESRGANer` 封装器完成，评测指标计算使用 `basicsr` 和 `lpips` 库。整体开发环境配置简洁，依赖项均为开源资源，适合在个人计算机上快速搭建与复现。

### 5.2 系统功能实现

#### 5.2.1 模型推理模块实现

后端推理模块的核心职责是将前端上传的低分辨率图像送入 Real-ESRGAN 预训练模型，完成超分辨率重建并返回结果。完整的数据流如下：(1) 前端通过 `/process` 接口以 `multipart/form-data` 格式上传图像字节流；(2) 后端利用 `cv2.imdecode` 解码为 BGR 格式的 NumPy 矩阵；(3) 将矩阵传入 `RealESRGANer.enhance()` 方法，由封装器内部完成预处理、模型前向推理和后处理，输出超分结果矩阵；(4) 结果经 `cv2.imencode` 编码为 PNG 字节流，再通过 Base64 编码嵌入 JSON 响应返回前端。

为避免模型重复加载，系统使用全局字典 `MODEL_CACHE` 缓存已初始化的模型实例。切换模型时，后端根据名称构造对应网络结构（`RealESRGAN_x4plus` 采用 RRDBNet，`realesr-general-x4v3` 采用 SRVGGNetCompact），从 `weights` 目录加载预训练权重并写入缓存。由于测试环境无独立 GPU，系统默认使用 CPU 推理（`gpu_id=None`）；若部署至 GPU 服务器，仅需调整 `gpu_id` 参数即可启用加速。

#### 5.2.2 单图像处理功能实现

单图像处理流程由前端页面与后端 `/process` 接口协同完成。用户选择图片后，前端通过 `FileReader` 预览原图，构建 `FormData` 请求并提交。后端返回超分结果图、拼接对比图、输入输出尺寸及模型名称后，前端刷新界面状态。结果数据均以 Base64 编码嵌入 JSON，确保传输兼容性。

对比图由后端 `make_compare_image()` 生成：将原图与超分结果左右拼接，标注尺寸信息；若原图需缩放对齐画布，采用 `cv2.INTER_CUBIC` 双三次插值，在清晰度与抗锯齿间取得平衡。前端对上传文件进行类型校验，仅接受 JPEG、PNG 等常见格式；若推理失败或返回错误状态码，前端会显示具体错误提示而非静默失败。界面采用左右双栏布局，原图下载按钮置于左栏，超分结果与对比图下载按钮置于右栏。系统单图像超分主界面如图5-1所示。

![图5-1 系统Web主界面（单图像超分功能展示）](images/fig5-1.png)

图5-1 系统Web主界面（单图像超分功能展示）

#### 5.2.3 批量处理与 ZIP 下载功能实现

批量处理是系统在单图像功能基础上的重要扩展。用户可一次性选择多张图片，通过“开始批量超分”统一提交至 `/batch_process` 接口。后端遍历所有文件，依次完成解码、推理、结果保存和对比图生成，并将结果与状态返回前端。处理结果同时缓存于内存和本地 `batch_outputs/<batch_id>/` 目录，供后续下载。

批量下载提供两个接口：`/batch_download_zip/<batch_id>/output` 和 `/batch_download_zip/<batch_id>/compare`。后端优先从内存缓存读取数据，利用 `BytesIO` 和 `zipfile` 在内存中动态生成 ZIP 包，以流式方式返回浏览器。该方案将打包与传输合为一体，减少了 OOM 风险，也规避了中文路径兼容性问题。当前内存缓存未设置自动过期清理，测试中单次批量限制在 50 张以内以保障稳定性。批量处理界面如图5-2所示。

![图5-2 系统Web批量处理界面展示](images/fig5-2.png)

图5-2 系统Web批量处理界面展示

#### 5.2.4 对比图与可视化展示实现

为便于效果对比，系统提供结果放大查看与对比图下载功能。后端统一排版的拼接对比图顶部含标题、底部标注分辨率。前端利用模态框组件弹窗展示高分辨率 PNG 原图，用户通过浏览器缩放观察细节，无需二次插值，所见即所得。这一设计兼顾了交互体验与论文展示需要。

#### 5.2.5 评测模块实现

实验评测模块由独立的 `evaluate.py` 脚本实现。其核心流程为：读取 HR 图像 → `mod_crop` 保证尺寸整除 → 双三次下采样生成 LR 并保存至 `LR_bicubic/X4` → 调用指定模型进行超分重建 → 计算 PSNR、SSIM 和 LPIPS → 导出逐图像 CSV 和统计图表。该模块支持 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3 四种模型，LPIPS 计算所需的感知模型权重已缓存至 `weights/torch_cache`。LR 图像默认复用已有结果，可通过 `--regenerate-lr` 参数强制重新生成，实现完整的实验闭环。

### 5.3 系统测试

系统功能测试采用黑盒测试方法，逐项验证需求文档中的核心功能。测试环境与开发环境一致，结果如表5-1所示。

表5-1 系统功能测试结果

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

测试结果表明，系统已完整覆盖毕业设计任务书中的主要功能需求，能够满足图像超分辨率处理、结果展示、批量导出和实验评测等应用场景。批量处理受限于 CPU 内存，单次建议不超过 50 张图片；ZIP 打包大小受磁盘空间限制；LPIPS 计算单张图耗时约 5–30 秒（取决于图像尺寸），但均可稳定完成。

### 5.4 算法测试

#### 5.4.1 测试数据集与对比模型

算法测试选用 Set5、Set14 和 BSD100 三个公开数据集。这些数据集是超分辨率研究中最通用的标准基准，涵盖人物、动物、建筑等场景。高分辨率 Ground Truth 存放于 `datasets/<dataset>/HR/`，经双三次下采样生成 ×4 低分辨率输入存入 `LR_bicubic/X4` 路径。对比模型选取 Bicubic（传统插值基线）、ESRGAN_x4（经典 GAN 模型）、RealESRGAN_x4plus（通用模型）和 realesr-general-x4v3（轻量模型），形成从传统方法到前沿 GAN 的完整对比链。

#### 5.4.2 评价指标

采用 PSNR、SSIM 和 LPIPS 三项指标，分别在像素级、结构级和感知级评估重建质量。实验固定 ×4 放大倍率，`crop_border=4`，在 Y 通道上计算 PSNR 和 SSIM。LPIPS 计算需启用 `--calc-lpips` 标志，首次运行会自动下载约 500MB 的预训练模型至 `weights/torch_cache`。

#### 5.4.3 定量实验结果

表5-2 不同模型在 Set5 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 27.6918 | 0.8102 | 0.3470 |
| ESRGAN_x4 | 17.2647 | 0.3670 | 0.5345 |
| RealESRGAN_x4plus | 25.4259 | 0.7851 | 0.1718 |
| realesr-general-x4v3 | 25.5326 | 0.7933 | 0.1806 |

表5-3 不同模型在 Set14 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 25.0055 | 0.7054 | 0.4603 |
| ESRGAN_x4 | 15.7702 | 0.2534 | 0.6229 |
| RealESRGAN_x4plus | 23.6790 | 0.6678 | 0.2453 |
| realesr-general-x4v3 | 23.7539 | 0.6827 | 0.2564 |

表5-4 不同模型在 BSD100 数据集上的测试结果

| 模型 | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Bicubic | 24.6329 | 0.6648 | 0.5285 |
| ESRGAN_x4 | 15.9418 | 0.2670 | 0.6021 |
| RealESRGAN_x4plus | 23.4753 | 0.6194 | 0.2858 |
| realesr-general-x4v3 | 23.7871 | 0.6369 | 0.3083 |

#### 5.4.4 实验结果综合分析

Bicubic 在三组数据集上均取得最高 PSNR，这是因为测试低分图像由双三次下采样生成，两者插值核同源，像素误差最小。但其 LPIPS 远高于 Real-ESRGAN 模型，暴露出平滑模糊的本质缺陷。这一现象恰好印证了感知-失真权衡理论：更低的 MSE 并不等同更优的视觉质量。

两种 Real-ESRGAN 模型的 LPIPS 显著优于 Bicubic 和 ESRGAN_x4，尤其在 Set5 上，x4plus 的 LPIPS 仅为 Bicubic 的一半（0.1718 vs 0.3470），表明其在感知层面更接近原始高清图像。realesr-general-x4v3 的 PSNR 和 SSIM 略高，x4plus 的 LPIPS 更低，反映了二者在保真度与感知质量上的不同侧重：轻量模型结构保守，像素偏差小；通用模型 GAN 成分更重，纹理更真实。

ESRGAN_x4 各项指标均偏低，原因在于其仅针对双三次退化训练且无高阶降质建模，对测试退化形式难以适应，易产生伪影。本实验中主要作为技术演进的历史参考。

#### 5.4.5 图表结果说明

实验生成了三张柱状图，分别展示各模型在三个数据集上的 PSNR、SSIM 和 LPIPS 对比。

![图5-3 不同模型在 Set5、Set14 和 BSD100 上的 PSNR 对比](images/fig5-3.png)

图5-3 不同模型在 Set5、Set14 和 BSD100 上的 PSNR 对比

![图5-4 不同模型在 Set5、Set14 和 BSD100 上的 SSIM 对比](images/fig5-4.png)

图5-4 不同模型在 Set5、Set14 和 BSD100 上的 SSIM 对比

![图5-5 不同模型在 Set5、Set14 和 BSD100 上的 LPIPS 对比（Lower Better）](images/fig5-5.png)

图5-5 不同模型在 Set5、Set14 和 BSD100 上的 LPIPS 对比（Lower Better）

图中蓝、橙、绿、紫分别对应 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3。柱状图直观地呈现出：Bicubic 在 PSNR 和 SSIM 上占优，Real-ESRGAN 两模型在 LPIPS 上优势明显，且 x4plus 感知质量最优。可视化结果与前述定量数据完全吻合。

### 5.5 本章小结

本章依次介绍了系统的开发环境、各功能模块的具体实现、系统功能测试结果以及算法测试的实验设计与分析。系统实现部分详细阐述了模型推理链路、单图像与批量处理流程、对比图生成机制和评测模块的构建细节，并辅以实际界面截图。测试结果表明系统功能完整、运行稳定；算法测试则通过三个数据集、四项指标的综合对比，验证了 Real-ESRGAN 模型在感知质量上的显著优越性，以及通用与轻量模型的互补特性。整体上，系统已达成原型设计目标，为真实场景图像增强提供了可用方案。

## 第6章 总结与展望

### 6.1 工作总结

本文围绕“基于 Real-ESRGAN 的图像超分辨率系统设计与实现”这一课题，完成了从理论梳理、需求分析、系统设计到功能实现与测试验证的完整研究流程。

在理论层面，本文系统梳理了图像超分辨率的基本概念与病态逆问题本质，深入分析了 Real-ESRGAN 的生成器结构、判别器设计、高阶降质建模策略与多损失联合优化机制。通过对 RRDB 模块的三层设计思想、U-Net 判别器的像素级反馈机制以及谱归一化的稳定性作用的阐述，为系统的模型选型和集成提供了理论支撑。同时，本文对 PSNR、SSIM 和 LPIPS 三类评价指标进行了对比分析，并引入感知-失真权衡理论，为实验部分的指标解读奠定了理论基础。

在系统层面，本文基于官方公开预训练模型完成了后端推理模块与 Web 可视化界面的构建。后端采用 Python 标准库搭建轻量级 HTTP 服务，结合 PyTorch 和 OpenCV 实现模型的加载、推理和结果编码；前端使用原生 HTML、CSS 和 JavaScript 实现了左右双栏布局的交互界面，支持单图像超分、批量处理、原图与结果对比展示、细节放大和 ZIP 批量导出等功能。系统整体依赖精简、部署便捷，适合在个人计算机上快速搭建与使用。

在实验层面，本文扩展实现了多模型评测模块，在 Set5、Set14 和 BSD100 三个公开数据集上完成了 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3 四种方法的定量对比。实验的核心发现在于清晰地展示了图像超分辨率任务中感知-失真权衡的具体表现：传统双三次插值在 PSNR 指标上领先，但其 LPIPS 远落后于 Real-ESRGAN 系列模型；RealESRGAN_x4plus 以微小的像素保真度代价换取了接近一倍的感知质量提升。此外，通用模型与轻量模型在保真度与感知质量上各有侧重，为不同部署场景提供了明确的模型选择依据。

在工程层面，本文构建的系统将 Real-ESRGAN 从学术代码封装为可交互的 Web 应用，证明了基于 Python 标准库的轻量级后端架构足以承担原型阶段的超分辨率服务任务。系统中实现的批量处理、内存缓存 ZIP 打包和评测自动化等模块，为同类图像增强系统的快速落地提供了可参考的实现方案。

### 6.2 展望

尽管本文完成了系统的设计、实现与实验验证，但仍存在若干不足，可作为后续研究的方向。

模型方面，本文使用的是官方预训练权重，在通用自然图像上表现良好，但未验证在垂直场景下的效果。后续可选取医学影像、低光监控图像或历史档案扫描件等典型场景，构建小规模标注数据集，采用 LoRA 等参数高效微调方法对模型进行适配，以进一步提升特定领域的重建质量。此外，随着扩散模型在图像复原领域的快速发展，可探索引入基于扩散模型的真实场景超分方案，以进一步改善感知质量。

评测方面，当前实验仅在 Set5、Set14 和 BSD100 三个合成退化数据集上进行。这些数据集的低分辨率输入均由标准双三次下采样生成，无法完全代表真实世界图像经历的复杂退化。后续可引入 RealSR、DRealSR 等真实场景超分基准，以更全面地评估模型在实际应用中的表现。

系统方面，当前的批量处理内存缓存机制未设置自动过期清理，长时间运行可能导致内存持续增长。后续可引入基于 LRU 策略的缓存淘汰机制，为缓存设置最大条目数，超出限制时自动清除较早的缓存数据。此外，当前系统主要面向本地单机部署，尚未集成用户管理、历史记录查询、任务队列管理等完整的应用层功能，后续可逐步完善以使系统更接近可交付的成熟产品形态。

## 参考文献

[1] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision (ECCV). Cham: Springer, 2014: 184-199.

[2] Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 4681-4690.

[3] Wang X, Yu K, Wu S, et al. ESRGAN: Enhanced super-resolution generative adversarial networks[C]//Proceedings of the European Conference on Computer Vision (ECCV) Workshops. 2018.

[4] Johnson J, Alahi A, Li F F. Perceptual losses for real-time style transfer and super-resolution[C]//European Conference on Computer Vision (ECCV). Cham: Springer, 2016: 694-711.

[5] Wang X, Xie L, Dong C, et al. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. 2021: 1905-1914.

[6] 刘东, 董超, 王兴刚, 等. Real-ESRGAN：真实世界图像超分辨率研究[J]. 计算机研究与发展, 2021.

[7] Blau Y, Michaeli T. The perception-distortion tradeoff[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 6228-6237.

[8] Zhang R, Isola P, Efros A A, et al. The unreasonable effectiveness of deep features as a perceptual metric[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 586-595.

[9] Horé A, Ziou D. Image quality metrics: PSNR vs. SSIM[C]//Proceedings of the International Conference on Pattern Recognition (ICPR). 2010: 2366-2369.

[10] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.

[11] 唐艳秋, 潘泓, 朱亚平, 等. 图像超分辨率重建研究综述[J]. 电子学报, 2020, 48(7): 1407-1420.

[12] 王睿琪. 图像超分辨率重建综述[J]. 计算机科学与应用, 2024, 14(2): 350-359.

[13] 钟文莉, 等. 深度学习图像超分辨率重建综述[J]. 软件导刊, 2025.

[14] 黄子琪, 等. 复杂退化模型下图像超分辨率算法综述[J]. 郑州大学学报(理学版), 2024.

[15] 薛鑫杰, 等. 基于深度学习的单幅图像超分辨率重建算法综述[J]. 黑龙江科学, 2023.

[16] Chen X, Wang X, Zhou J, et al. Activating more pixels in image super-resolution transformer[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023.

[17] Wu R, Yang T, Sun L, et al. SeeSR: Towards semantics-aware real-world image super-resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024.

[18] Lin X, He J, Chen Z, et al. DiffBIR: Towards blind image restoration with generative diffusion prior[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2024.

[19] Sun L, Wu R, Ma Z, et al. Pixel-level and semantic-level adjustable super-resolution: A dual-LoRA approach[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2025.

[20] Chen B, Li G, Wu R, et al. Adversarial diffusion compression for real-world image super-resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2025.

[21] 陈文达, 等. 基于生成对抗网络的图像超分辨率重建综述[J]. 计算机应用, 2024.

## 致谢

在本次毕业设计与论文撰写过程中，我系统梳理了图像超分辨率相关理论，并结合项目代码完成了 Real-ESRGAN 超分辨率系统的设计、实现与测试。在此，谨向指导教师在课题选题、系统实现和论文修改过程中给予的耐心指导与帮助表示衷心感谢。感谢开源社区提供 Real-ESRGAN 相关基础代码与公开预训练权重，感谢公开数据集和相关论文作者为模型测试与性能分析提供了重要参考。通过本次毕业设计，我进一步加深了对图像超分辨率与深度学习方法的理解，并在系统集成、实验评测和学术写作等方面获得了较为全面的锻炼。

## 附录A 论文插图与结果文件说明

图2-1、图2-2为模型结构示意图，需使用 draw.io 绘制并导出为 PNG。图5-1、图5-2为系统运行截图。图5-3至图5-5由 `plot_evaluation_results.py` 脚本生成。各模型对比 CSV 文件保存在 `evaluation_results/` 目录下的对应数据集子目录中。

## 附录B 测试命令说明

```powershell
# Set5
python .\evaluate.py --hr-dir .\datasets\Set5\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set5_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

# Set14
python .\evaluate.py --hr-dir .\datasets\Set14\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set14_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

# BSD100
python .\evaluate.py --hr-dir .\datasets\Bsd100\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Bsd100_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

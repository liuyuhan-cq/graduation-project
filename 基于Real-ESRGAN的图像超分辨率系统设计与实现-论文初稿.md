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

该问题的核心困难在于其具有典型的不适定性。从信息论角度看，低分辨率图像的信息量远小于高分辨率图像，降质过程本质上是信息丢失，而逆映射是一对多的：同一个低分辨率观测可能对应无穷多种高分辨率重建结果。这意味着即使 $I_{LR}$ 被完美重建为某种高分辨率形式，也无法保证该结果与真实的 $I_{HR}$ 一致。因此，如何从有限的观测中恢复既满足数据一致性又具有感知真实性的细节，成为超分辨率研究的根本挑战[1-3]。

### 2.2 Real-ESRGAN 模型原理

#### 2.2.1 生成器结构

Real-ESRGAN 的生成器沿用了 ESRGAN 中提出的 RRDB（Residual-in-Residual Dense Block）结构。该结构的核心设计思想体现在三个层面：其一，残差中嵌套残差，使网络可以在不同粒度上学习恒等映射偏差，而非直接预测全部像素；其二，密集连接使每一层都能直接接收前面所有层的特征，增强了特征的复用效率和梯度流动；其三，移除了传统残差块中常见的批归一化（Batch Normalization）层——ESRGAN 原论文的消融实验表明，BN 层在深层 GAN 训练中会产生伪影，移除后可获得更干净的高频纹理[3,5]。这三项设计共同提高了网络的表达能力和训练稳定性，使得 RRDB 在纹理细节恢复方面较一般深层卷积网络更具优势。

在本文所集成的模型中，`RealESRGAN_x4plus` 采用 23 个 RRDB 模块，面向通用场景追求最高的重建质量；`RealESRGAN_x4plus_anime_6B` 采用 6 个 RRDB 模块，在保持对动漫图像适配能力的同时减小了模型规模；`realesr-general-x4v3` 则采用更轻量的 `SRVGGNetCompact` 结构，以进一步降低推理延迟和内存占用。在实际部署中，可根据应用场景（高画质/动漫优化/实时处理）灵活选择对应的模型版本。

#### 2.2.2 高阶降质建模

Real-ESRGAN 的重要改进之一在于提出了更贴近真实场景的高阶降质建模策略[5-6]。不同于传统方法通常仅考虑单次模糊与一次双三次下采样，该方法通过多轮随机模糊、缩放、噪声扰动和 JPEG 压缩的叠加，模拟真实图像在多次采集、传输和再压缩等过程中可能经历的复杂退化。这一策略的核心价值在于：通过在训练阶段极大扩展降质分布的覆盖范围，缩小了合成训练数据与真实测试输入之间的分布差异，从而在不依赖任何真实世界配对训练数据的前提下，使模型获得了对未知真实退化的泛化能力。Real-ESRGAN 原论文在多个真实场景测试集上的实验已验证了该策略的有效性[5]。需要指出的是，该高阶降质建模策略已在预训练模型训练阶段完成，系统推理时无需显式实现降质过程，而是直接调用预训练好的模型对输入图像进行超分辨率处理。

#### 2.2.3 判别器结构

在判别器设计方面，Real-ESRGAN 相较于 ESRGAN 常用的 PatchGAN 判别器，进一步引入了带谱归一化的 U-Net 判别器[5]。选择 U-Net 结构的主要原因在于：PatchGAN 判别器仅输出整体图像的单一真假判断，无法为生成器提供像素级的局部反馈；而 U-Net 判别器在每个空间位置都产生判别信号，能够对图像的局部纹理进行精细监督，有助于生成器在细节处产生更真实的纹理。

谱归一化方面，其原理是通过约束判别器中每一层的谱范数，将判别器的 Lipschitz 常数限制在一定范围内，从而防止判别器参数在对抗训练中过度增长，有效缓解训练过程中的梯度爆炸和振荡问题[5]。这一设计使得 Real-ESRGAN 在更大的退化空间上仍能稳定训练，并最终输出结构一致性和纹理真实性更优的重建结果。

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

## 第3章 系统需求分析与总体设计

### 3.1 系统需求分析

#### 3.1.1 功能需求

1. 支持用户上传单张图像并完成超分辨率重建。  
2. 支持多种模型切换，包括通用模型、动漫模型和轻量模型。  
3. 支持原图与超分结果以左右双栏形式同步展示，并标注输入与输出分辨率。  
4. 支持结果图像放大查看，便于观察局部细节。  
5. 支持批量图像处理，降低重复操作成本。  
6. 支持单张结果下载，以及批量结果和批量对比图以 ZIP 格式一键打包下载。  
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

单图像超分业务流程为：图像上传与前端预览 → 模型选择 → 后端推理 → 结果展示与下载。  
批量处理业务流程为：多图选择 → 批量提交 → 逐张推理与结果缓存 → 单张下载与 ZIP 批量下载。  
评测业务流程为：读取 HR 图像 → 生成或复用 LR 图像 → 模型推理 → 计算 PSNR、SSIM 和 LPIPS → 保存 SR 图像与对比图 → 导出统计表和柱状图。

## 第4章 系统设计与实现

### 4.1 开发环境与运行环境

系统开发环境为 Windows、Python 3.10、PyTorch 和 OpenCV，前端采用原生 HTML、CSS 和 JavaScript 实现，后端基于 Python 标准库构建轻量级 HTTP 服务。模型推理依赖 `RealESRGANer` 封装器，图像评测基于 `basicsr.metrics` 中的 PSNR 和 SSIM 计算函数，并结合 `lpips` 库计算 LPIPS 指标。整体环境配置较为简洁，适合在个人计算机上部署与演示。

### 4.2 模型推理模块实现

后端推理模块位于 `web/web_server.py`，其核心职责是将前端上传的低分辨率图像送入 Real-ESRGAN 预训练模型，完成超分辨率重建并返回结果。完整的数据流如下：

1. 前端通过 `/process` 接口以 `multipart/form-data` 格式上传原始图像字节流。  
2. 后端利用 `cv2.imdecode` 将字节流解码为 NumPy 矩阵（BGR 格式）。  
3. 将矩阵传入 `RealESRGANer.enhance()` 方法，由封装器内部完成预处理、模型前向推理和后处理，输出超分辨率重建后的图像矩阵。  
4. 结果矩阵经 `cv2.imencode` 以 PNG 格式编码为字节流，再通过 Base64 编码嵌入 JSON 响应返回前端。

为避免模型在多次请求中被重复加载，系统采用全局字典 `MODEL_CACHE` 缓存已初始化的模型实例。当用户切换模型时，后端根据模型名称构造对应网络结构（`RealESRGAN_x4plus` 采用 RRDBNet，`realesr-general-x4v3` 采用 SRVGGNetCompact），从 `weights` 目录加载对应的预训练权重并写入缓存。由于开发与测试环境未配置独立 GPU，系统默认使用 CPU 推理（`gpu_id=None`）。若部署至配有 GPU 的服务器，仅需将 `gpu_id` 设置为对应设备编号即可启用 GPU 加速，无需修改推理逻辑。

### 4.3 单图像处理功能实现

单图像处理功能由前端页面与后端推理接口协同完成。用户选择图片后，前端通过 `FileReader` 完成原图预览，并构造 `FormData` 请求发送至 `/process` 接口。后端完成推理后，返回超分结果图、拼接对比图、输入尺寸、输出尺寸以及模型名称等信息，前端据此更新页面状态并显示处理结果。超分结果与对比图经 Base64 编码后以 JSON 格式返回，确保了数据传输的兼容性。

对比图由后端 `make_compare_image()` 函数生成：将原图与超分结果左右拼接，并标注输入/输出尺寸信息。拼接前，原图部分若需缩放以对齐对比画布，采用 `cv2.INTER_CUBIC`（双三次插值）进行缩放。选择该算法的原因在于：双三次插值在清晰度和抗锯齿之间取得了较好平衡，适合视觉展示场景。

在异常处理方面，前端对用户上传的文件进行了类型校验，仅接受常见图像格式（JPEG、PNG 等）；若后端推理失败或返回错误状态码，前端会向用户显示具体的错误提示信息，而非静默失败。这一设计提升了系统的健壮性和用户体验。

在界面设计方面，系统采用左右双栏布局展示原图与超分结果，使用户能够直观比较前后差异。原图下载按钮设置于左侧区域，超分结果和对比图下载按钮设置于右侧区域，从而使界面逻辑更加清晰。

### 4.4 批量处理与 ZIP 下载功能实现

批量处理功能是本系统的重要扩展内容。用户点击“批量处理”后，可一次性选择多张图片，并通过“开始批量超分”将其统一提交至 `/batch_process` 接口。后端在接收到批量请求后，依次完成图像解码、模型推理、结果保存和对比图生成，并将每张图像的处理结果组织后返回前端。与此同时，系统将批处理结果缓存在内存与本地目录 `batch_outputs/<batch_id>/` 中，以支持后续下载操作。

针对批量结果下载需求，系统进一步设计了 `/batch_download_zip/<batch_id>/output` 和 `/batch_download_zip/<batch_id>/compare` 两类接口。当前端发起下载请求时，后端优先从内存缓存中读取结果数据，利用 `BytesIO` 和 `zipfile` 在内存中动态构建 ZIP 压缩包，再以流式方式返回浏览器。该方案将 ZIP 的构建与传输合为一步，避免了大文件在磁盘和内存间的多次搬运，从而减少了 OOM（内存溢出）的风险，也有效规避了中文路径环境下可能出现的兼容性问题。

需要指出的是，当前内存缓存未设置自动过期或容量上限，长时间连续批处理可能累积占用内存。在生产环境中，可引入基于 LRU（最近最少使用）策略的缓存淘汰机制，或根据系统负载自动调整缓存容量，以保障长时运行的稳定性。在本系统的测试场景中，单次批量通常不超过 50 张图像，该限制可有效避免内存溢出的发生。

### 4.5 对比图与可视化展示实现

为了便于分析超分效果，系统提供了结果放大查看与对比图下载功能。后端通过对原图与结果图进行统一排版，生成包含标题与尺寸信息的拼接对比图。前端则借助模态框组件实现大图弹窗预览：用户点击原图或结果图后，浏览器将已生成的高分辨率 PNG 图像在弹窗中直接展示，用户可通过浏览器的缩放功能观察局部细节。该设计无需对图像进行二次放大的插值处理，保证了用户所查看的细节与下载结果完全一致，增强了系统的交互体验，也便于实验结果在论文中的整理与展示。

### 4.6 评测模块实现

实验评测模块由 `evaluate.py` 实现，旨在对多个超分辨率模型在统一标准下进行定量对比。评测使用的测试数据来源于 Set5、Set14 和 BSD100 三个公开数据集，它们涵盖人物、动物、自然纹理及复杂场景，是超分辨率领域广泛采用的标准基准。

该模块的主要评测流程如下：首先读取高分辨率测试图像（Ground Truth）；然后对图像进行 `mod_crop` 处理，以保证尺寸可被放大倍率整除；接着采用双三次下采样生成对应低分辨率图像，并保存至 `LR_bicubic/X4` 目录；随后调用指定模型对低分辨率输入进行超分辨率重建；最后计算 PSNR、SSIM 和 LPIPS 等评价指标，并导出逐图像结果及统计文件。

该评测模块支持 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3 四种模型。为保证 LPIPS 在本地环境中的稳定计算，系统对感知模型权重的缓存路径进行了统一处理，从而避免因运行环境差异导致的下载或读写问题。在低分辨率图像生成方面，脚本默认将 LR 图像保存至指定目录；若此前已生成且尺寸匹配的 LR 图像，将直接复用，否则将重新生成；如需强制覆盖已有 LR 图像，可在运行评测时添加 `--regenerate-lr` 参数。通过该模块，系统形成了从测试输入生成、模型推理、指标计算到结果导出的完整实验闭环。

### 4.7 关键实现说明

系统实现中，模型初始化、单图像处理、批量任务调度、压缩包导出、低分辨率测试图像生成以及多模型评测等功能分别进行了模块化封装。具体而言，后端通过统一的模型初始化机制完成不同网络结构与预训练权重的匹配，并结合缓存策略减少重复加载开销；图像处理流程被抽象为可复用的推理接口，从而为单图像处理与批量处理提供统一支撑；批量任务管理模块负责组织多图输入、结果缓存与 ZIP 导出；评测部分则实现了测试数据准备、逐模型推理、指标统计与结果写出。上述设计保证了系统在功能扩展、实验复现和代码维护方面具有较好的可操作性。

通过上述实现，本文不仅完成了超分辨率系统原型的构建，也形成了较为完整的评测与分析流程。

## 第5章 系统测试与实验结果分析

### 5.1 实验目的与实验环境

本章实验主要验证以下三个方面：其一，系统能否稳定完成超分辨率重建与结果展示；其二，不同模型在公开数据集上的客观失真指标与感知质量指标表现；其三，本文扩展实现的批处理、结果导出和图表生成功能能否满足论文分析需求。

实验在本地 Windows 平台上完成。硬件环境为 AMD Ryzen 7 5800H with Radeon Graphics 处理器，未配备独立 GPU；操作系统为 Windows 11。软件环境包括 Python 3.10、PyTorch 1.13.1（CPU 版本）和 lpips 0.1.4。所有模型推理均在 CPU 上进行。

实验脚本 `evaluate.py` 负责完整的评测流水线：读取测试图像、生成低分辨率输入、调度各模型推理、计算三项指标、输出 CSV 文件。绘图脚本 `plot_evaluation_results.py` 读取评测输出的 CSV 文件，生成 PSNR、SSIM 和 LPIPS 三张柱状图。两个脚本通过 CSV 文件衔接，评测结果可独立于绘图脚本使用，便于复现和自定义分析。

### 5.2 数据集与测试方案

#### 5.2.1 测试数据集

本文选用 Set5、Set14 和 BSD100 三个公开数据集进行实验。选择这三组数据集的主要依据在于：它们涵盖人物、动物、自然纹理及复杂场景等多种图像内容，且是单图像超分辨率研究中最为通用的标准基准，几乎被所有主流超分论文采用。使用这些数据集，有助于将本文结果与已有文献进行横向参照。测试过程中，`datasets/<dataset>/HR/` 用于存放高分辨率 Ground Truth 图像，并以其为基准经双三次下采样生成对应低分辨率输入，保存至 `LR_bicubic/X4` 路径。所有模型共用同一组低分辨率输入，以保证实验的公平性与可比性。

#### 5.2.2 对比模型

本文选取以下四种方法进行比较。这四种模型构成了一条从传统方法到最新GAN方法的完备对比链：Bicubic 代表传统插值基线，ESRGAN 代表第一代感知驱动型GAN模型，RealESRGAN_x4plus 和 realesr-general-x4v3 则分别代表当前先进的通用模型和轻量模型。通过这一递进式对比，可以同时评估技术的代际演进（传统→GAN→高阶退化GAN）和模型的规模权衡（通用←→轻量）。

1. Bicubic：双三次插值基线，用于反映传统插值方法的性能。  
2. ESRGAN_x4：经典感知型 GAN 超分辨率模型。  
3. RealESRGAN_x4plus：Real-ESRGAN 通用模型。  
4. realesr-general-x4v3：轻量化 Real-ESRGAN 通用模型。

#### 5.2.3 评价指标

实验采用 PSNR、SSIM 和 LPIPS 三项指标，从三个互补维度对模型进行评估：PSNR 衡量像素级重建精度，SSIM 衡量结构相似性，LPIPS 衡量视觉感知质量（该指标越低越好）。三项指标联合使用，可避免单一指标的片面性——例如，一个模型可能在 PSNR 上表现优异但感知质量较差，反之亦然。实验放大倍率固定为 ×4，`crop_border = 4`，并在 Y 通道上计算 PSNR 和 SSIM。LPIPS 指标的计算需启用 `--calc-lpips` 标志，首次运行会自动下载约 500MB 的预训练感知模型并缓存至本地 `weights/torch_cache` 目录。

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

在 BSD100 数据集上，realesr-general-x4v3 在 PSNR 和 SSIM 上略高于 RealESRGAN_x4plus，而后者在 LPIPS 上更低。该结果说明两者在保真度与感知质量上的表现具有一定互补性。从三组数据集的整体趋势来看，随着数据集规模从 Set5（5张）增加到 BSD100（100张），所有模型的绝对指标均有所下降，但模型间的相对排序保持稳定，说明结论具有良好的跨数据集一致性。

### 5.4 实验结果综合分析

#### 5.4.1 失真指标与感知指标的权衡

Bicubic 在三组数据集上的 PSNR 均取得最高值，主要原因在于本文测试所用的低分辨率图像由双三次下采样生成，因此 Bicubic 插值与测试降质过程在数学上高度一致——两者使用的是同一插值核。这种“同源匹配”使得其像素级误差最小。然而，从 LPIPS 指标来看，Bicubic 的结果明显劣于 Real-ESRGAN，表明其重建图像虽然在数值上更接近参考图像，但视觉上往往更加平滑、纹理不足。这一现象与感知—失真权衡理论[7]一致，即较低的失真并不必然对应更优的感知质量。

从模型设计角度看，Real-ESRGAN 系列之所以在 PSNR 上不及 Bicubic，与其训练目标密切相关：GAN 损失使生成器倾向于产生高频纹理，而这些纹理虽然在视觉上更真实，但在像素位置上与 Ground Truth 并非严格对齐，因此被 MSE 惩罚。这并非模型的缺陷，而是感知优化方向的合理体现。

#### 5.4.2 LPIPS 指标的指导意义

实验结果表明，RealESRGAN_x4plus 与 realesr-general-x4v3 在三组数据集上的 LPIPS 均明显优于 Bicubic 和 ESRGAN_x4。由此可见，仅依赖 PSNR 和 SSIM 难以全面反映真实场景超分模型的优势。以 Set5 为例，RealESRGAN_x4plus 的 LPIPS（0.1718）仅为 Bicubic（0.3470）的一半左右，意味着在深度网络的特征空间中，其重建结果与原始图像的距离远小于插值结果。这一显著差异说明，对于以感知质量为目标的超分辨率任务，LPIPS 相较于传统失真指标具有不可替代的参考价值。

#### 5.4.3 两种 Real-ESRGAN 模型的差异

实验结果显示，realesr-general-x4v3 在 PSNR 和 SSIM 上略占优势，而 RealESRGAN_x4plus 的 LPIPS 更低。这一差异可以从模型架构和设计目标中得到解释：RealESRGAN_x4plus 采用 23 个 RRDB 模块的深层结构，配合更强的 GAN 训练，倾向于生成更具感知真实感的细节；realesr-general-x4v3 采用轻量的 SRVGGNetCompact 结构，参数更少，生成纹理的能力相对保守，因此在像素位置的“犯错”更少，PSNR 反而更高。这一权衡关系说明，在实际部署中可根据应用场景选择模型：若追求最优画质，选用 x4plus；若需在推理速度和保真度间取得平衡，则 x4v3 更为合适。

#### 5.4.4 ESRGAN_x4 指标偏低的原因

ESRGAN_x4 在三组数据集上的各项指标均明显偏低。这一结果可以从两方面解释。其一，ESRGAN 的训练退化模型仅为简单的双三次下采样，而本实验虽然测试输入也是双三次退化，但 ESRGAN 的感知优化方向使其在简单退化上也可能产生与 Ground Truth 差异较大的纹理。其二，ESRGAN 缺少 Real-ESRGAN 中引入的高阶退化建模和 U-Net 谱归一化判别器，其对退化分布外图像的泛化能力和训练稳定性均不及 Real-ESRGAN。因此，本实验中 ESRGAN 主要作为体现技术演进的历史基线，而非当前部署的候选模型。

### 5.5 图表结果说明

本文生成了三张可直接用于论文排版的柱状图：

图5-1 不同模型在 Set5、Set14 和 BSD100 上的 PSNR 对比  
图5-2 不同模型在 Set5、Set14 和 BSD100 上的 SSIM 对比  
图5-3 不同模型在 Set5、Set14 和 BSD100 上的 LPIPS 对比（Lower Better）

图中蓝、橙、绿、紫四种颜色分别对应 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3。通过柱状图可以直观观察到两大趋势：在 PSNR 和 SSIM 方面，Bicubic 在各数据集上均领先，Real-ESRGAN 两模型紧随其后；在 LPIPS 方面，Real-ESRGAN 两模型在各数据集上均大幅领先，且 x4plus 整体感知质量最优。这些可视化结果与 5.3 节的定量数据完全对应。

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

测试结果表明，系统已完整覆盖毕业设计任务书中的主要功能需求，能够满足图像超分辨率处理、结果展示、批量导出和实验评测等应用场景。需要说明的是，批量处理受限于 CPU 内存，单次建议不超过 50 张图片；ZIP 打包大小受磁盘空间限制；LPIPS 计算单张图耗时约 5–30 秒（取决于图像尺寸），但均可稳定完成。

### 5.7 本章小结

本章从数据集与测试方案、定量结果、图表输出以及系统功能测试等方面对本文所实现的系统进行了验证。实验结果表明，系统能够稳定完成超分辨率重建与多模型评测任务；Real-ESRGAN 模型在 LPIPS 指标上明显优于传统插值方法和上一代 GAN 方法，且两种 Real-ESRGAN 变体在保真度与感知质量上各有侧重，为不同应用场景提供了灵活选择；系统前后端交互、批量处理和结果导出功能运行稳定，达到了毕业设计中“系统原型实现与实验验证”的目标。

## 第6章 总结与展望

### 6.1 工作总结

本文围绕“基于 Real-ESRGAN 的图像超分辨率系统设计与实现”展开研究。在理论层面，梳理了图像超分辨率的基本概念与病态逆问题本质，分析了 Real-ESRGAN 的生成器结构、判别器设计、高阶降质建模策略与多损失联合优化机制。在系统层面，基于官方公开预训练模型完成了后端推理模块与 Web 可视化界面的构建，实现了单图像处理、批量处理、原图与结果对比展示、细节放大和 ZIP 批量导出等功能。在实验层面，扩展实现了多模型评测模块，在 Set5、Set14 和 BSD100 三个公开数据集上完成了 Bicubic、ESRGAN_x4、RealESRGAN_x4plus 和 realesr-general-x4v3 四种方法的定量对比。

本文实验的核心发现在于清晰地展示了图像超分辨率任务中“感知—失真权衡”的具体表现：传统双三次插值在 PSNR 指标上领先，但其 LPIPS 远落后于 Real-ESRGAN 系列模型；RealESRGAN_x4plus 以微小的像素保真度代价换取了接近一倍的感知质量提升（在 Set5 上 LPIPS 从 0.3470 降至 0.1718）。此外，通用模型与轻量模型在保真度与感知质量上各有侧重，为不同部署场景提供了明确的模型选择依据。

在工程层面，本文构建的系统将 Real-ESRGAN 从学术代码封装为可交互的 Web 应用，证明了基于 Python 标准库的轻量级后端架构足以承担原型阶段的超分辨率服务任务。系统中实现的批量处理、内存缓存 ZIP 打包和评测自动化等模块，为同类图像增强系统的快速落地提供了可参考的实现方案。

### 6.2 主要成果与创新点

1. 构建了从模型调用到界面交互的完整超分辨率系统，实现了单图像与批量图像处理全流程。  
2. 在系统中实现了批量结果和批量对比图的 ZIP 一键下载功能，增强了工程实用性。  
3. 设计并实现了多模型评测脚本，引入 LPIPS 指标，将分析维度由传统保真度评价拓展到感知质量评价。  
4. 在 Set5、Set14 和 BSD100 数据集上完成四模型对比实验，量化展示了感知—失真权衡特征。  
5. 形成了一套与项目代码一致的论文实验流程，可为毕业答辩展示及后续优化提供支撑。

### 6.3 不足与展望

尽管本文完成了系统设计、实现与实验验证，但仍存在以下不足，可作为后续研究的方向。

**不足一：模型未针对特定场景进行微调。** 本文使用的是官方预训练权重，在通用自然图像上表现良好，但未验证在垂直场景（如医学影像、低光监控图像、历史档案扫描件）下的效果。后续可选取一至两个典型场景，构建小规模标注数据集，以 LoRA 等参数高效微调方法对模型进行适配，并对比微调前后的 PSNR 和 LPIPS 增益[19]。

**不足二：评测数据集覆盖范围有限。** 当前实验仅在 Set5、Set14 和 BSD100 三个合成退化数据集上进行。这些数据集虽然广泛使用，但其低分辨率输入均由标准双三次下采样生成，无法完全代表真实世界图像的复杂退化。后续可引入 RealSR、DRealSR 等真实场景超分基准，以更全面地评估模型在实际应用中的表现。

**不足三：批量处理的内存缓存机制有待完善。** 当前系统的批处理结果缓存至内存且未设置过期清理，长时间运行可能导致内存持续增长。后续可引入基于 LRU（最近最少使用）策略的缓存淘汰机制，为缓存设置最大条目数（如保留最近 10 个批次的处理结果），超出限制时自动清除较早的缓存数据，从而保障系统在长时间运行下的稳定性。

**不足四：系统功能仍有扩展空间。** 当前系统聚焦于图像超分辨率的核心处理与评测流程，尚未集成用户管理、历史记录查询、任务队列管理等完整的应用层功能。这些功能在后续可逐步完善，以使系统更接近可交付的成熟产品形态。此外，可探索引入基于扩散模型的真实场景超分方案[18][20]，以进一步改善感知质量。

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

图5-1、图5-2、图5-3以及Set5/Set14/BSD100各模型对比CSV文件，均保存在`plots_4models_lpips` 目录及对应数据集子目录中，具体路径见项目 `evaluation_results/` 目录。

## 附录B 测试命令说明

```powershell
# Set5
python .\evaluate.py --hr-dir .\datasets\Set5\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set5_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

# Set14
python .\evaluate.py --hr-dir .\datasets\Set14\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Set14_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

# BSD100
python .\evaluate.py --hr-dir .\datasets\Bsd100\HR --models Bicubic ESRGAN_x4 RealESRGAN_x4plus realesr-general-x4v3 --output .\evaluation_results\Bsd100_4models_lpips --scale 4 --crop-border 4 --test-y-channel --calc-lpips --report-psnr-floor 10

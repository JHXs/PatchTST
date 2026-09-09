# TransFusion 中央式两步复现约定

日期：2026-09-09

用途：为后续独立实验分支冻结 TransFusion 强对照的数学目标、超参数映射、截距口径和数值验收条件。本文件不修改候选机制，不读取真实数据，也不授权训练。

## 1. 权威来源与符号

唯一算法来源为 He 等的 [TransFusion: Covariate-Shift Robust Transfer Learning for High-Dimensional Regression](https://proceedings.mlr.press/v238/he24a.html)、[PMLR PDF](https://proceedings.mlr.press/v238/he24a/he24a.pdf) 及同稿 [arXiv:2404.01153](https://arxiv.org/abs/2404.01153)，AISTATS 2024，PMLR 238:703–711。以下中央式目标转录自原文第 3–4 页的式（2）和式（5），不使用 D-TransFusion。

沿用论文符号：目标任务为 $k=0$，有 $n_T$ 个样本；$K$ 个源任务各有 $n_S$ 个样本；$X^{(k)}\in\mathbb{R}^{n_k\times p}$、$y^{(k)}\in\mathbb{R}^{n_k}$、$\beta^{(k)}\in\mathbb{R}^p$。论文假设源样本量相同，

\[
N=n_T+Kn_S. \tag{0}
\]

项目 L0 中以 $n_0$ 记 $n_T$，以 $S$ 记论文的 $K$。不要把项目的站点偏差系数 $a_s$ 与 TransFusion 的 fused 权重混用；实现和日志统一把后者命名为 `tf_fused_weight`。

## 2. 第一步：原文式（2）及聚合量

论文的中央式 co-training 优化问题为

\[
\widehat{\boldsymbol\beta}\in
\arg\min_{\boldsymbol\beta\in\mathbb{R}^{(K+1)p}}
\left\{
\frac{1}{2N}\sum_{k=0}^{K}
\left\|y^{(k)}-X^{(k)}\beta^{(k)}\right\|_2^2
+\lambda_0\left(
\left\|\beta^{(0)}\right\|_1
+\sum_{k=1}^{K}a_k
\left\|\beta^{(k)}-\beta^{(0)}\right\|_1
\right)
\right\}. \tag{TF-2}
\]

分母必须是总样本量 $2N$。展开后，每个目标或源观测的平方残差系数均为 $1/(2N)$；不能先把每个任务的损失除以自己的 $n_k$ 再求平均。正则部分的有效系数为：目标系数 $\beta^{(0)}$ 的稀疏惩罚是 $\lambda_0$，第 $k$ 个参数差 $\beta^{(k)}-\beta^{(0)}$ 的 fused 惩罚是 $\lambda_0a_k$。原式没有单独的 $\|\beta^{(k)}\|_1$ 项。

得到各任务解后，论文先构造样本量加权平均

\[
\widehat w=
\sum_{k=0}^{K}\frac{n_k}{N}\widehat\beta^{(k)}
=\frac{n_S}{N}\sum_{k=1}^{K}\widehat\beta^{(k)}
+\frac{n_T}{N}\widehat\beta^{(0)},
\qquad n_0=n_T,\ n_k=n_S\ (k\geq1). \tag{TF-w}
\]

L0 的五个源站均有 $n_S=768$。人工少标签组 $n_0=96$ 时，$N=3936$，目标权重为 $96/3936$，每个源任务权重为 $768/3936$；全标签压力对照 $n_0=768$ 时，$N=4608$，六个任务权重均为 $1/6$。中间量是所有任务估计的样本量加权和，不是直接取 $\widehat\beta^{(0)}$，也不得用等任务权重替换式（TF-w）。

## 3. 第二步：原文式（5）的目标残差回归

论文的 local debias 不是重新拟合完整目标模型，而是在目标数据上对第一步残差拟合稀疏增量：

\[
\widehat\delta\in\arg\min_{\delta\in\mathbb{R}^p}
\left\{
\frac{1}{2n_T}
\left\|y^{(0)}-X^{(0)}\widehat w-X^{(0)}\delta\right\|_2^2
+\widetilde\lambda\|\delta\|_1
\right\}, \tag{TF-5a}
\]

\[
\widehat\beta^{(0)}_{\mathrm{TransFusion}}
=\widehat w+\widehat\delta. \tag{TF-5b}
\]

因此实现应先固定 $r^{(0)}=y^{(0)}-X^{(0)}\widehat w$，再解以 $X^{(0)}$ 为设计矩阵、以 $r^{(0)}$ 为响应的 Lasso。第二步每个目标观测的平方残差系数是 $1/(2n_T)$，不是 $1/(2N)$。最终验证和测试预测使用 $X\widehat\beta^{(0)}_{\mathrm{TransFusion}}$。

## 4. Fused 系数和共同 12 点网格

式（TF-2）本身把 $\lambda_0$ 和 $a_k$ 留作调节量。论文定理 2 定义事件

\[
\mathcal A=\left\{
\frac{s\log p}{n_S}
\geq \bar h\sqrt{\frac{\log p}{n_T}}
\right\}, \tag{TF-A}
\]

并给出分支式选择

\[
\lambda_0=c_0\left(
\sqrt{\frac{\log p}{N}}\,\mathbf 1_{\mathcal A}
+\sqrt{\frac{\log p}{n_S}}\,\mathbf 1_{\mathcal A^c}
\right), \tag{TF-lambda0}
\]

\[
a_k=8\left(
\sqrt{\frac{n_S}{N}}\,\mathbf 1_{\mathcal A}
+\frac{n_S}{N}\,\mathbf 1_{\mathcal A^c}
\right),
\qquad
\widetilde\lambda=c_1\sqrt{\frac{\log p}{n_T}}. \tag{TF-ak}
\]

合成 L0 不知道 $s$ 和 $\bar h$ 的 oracle 分支，也不为 TransFusion 增加二维搜索预算。项目复现固定采用上式的 $\mathcal A$ 分支，并把论文中可分别取值的 $c_0,c_1$ 约束成同一个验证乘数 $c$。这是一项预注册复现约定，不应写成论文原文的唯一实现：

\[
C=\left\{10^{-2+3j/11}:j=0,1,\ldots,11\right\}, \tag{G-1}
\]

\[
\lambda_0(c)=c\sqrt{\frac{\log p}{N}},
\qquad
a_k=8\sqrt{\frac{n_S}{N}},
\qquad
\widetilde\lambda(c)=c\sqrt{\frac{\log p}{n_0}}. \tag{G-2}
\]

这里 `log` 是自然对数。每个 $c$ 只产生一个完整的两步 TransFusion 候选，共 12 个候选；第一步和第二步不得分别搜索不同的 $c$。按 target validation MSE 选取最小者，若差值不超过 $10^{-12}$，选择较大的 $c$。选定后直接评估保留的 test，不把 validation 合并回训练，以免改变 $n_0$、$N$ 和预注册权重。$n_0=96$ 与 $n_0=768$ 两组分别重新计算式（G-2），不能复用数值正则系数。

## 5. 截距与预处理口径

原文的线性模型和式（TF-2）、（TF-5a）均没有截距变量；论文假设设计向量均值为零。故 `paper_exact` 主对照必须满足：

- 求解器关闭隐式 intercept；
- 不添加常数列；
- 不进行逐任务 $y$ 去均值，因为这会先验移除任务间均值差；
- 只接收 L0 协议已经冻结的输入矩阵，优化器内部不得再次逐任务标准化。

在这个主对照中不存在“截距是否惩罚”的选择，因为根本没有截距参数。按照 L0 公平对照协议，还必须单独实现 `tf_unpenalized_task_intercepts`：第一步为每个任务加入不受惩罚的截距 $\alpha_k$，所有 ℓ1/fused 惩罚仍只作用于斜率；聚合时同时计算 $\widehat w_\alpha=\sum_{k=0}^{K}(n_k/N)\widehat\alpha_k$；第二步再拟合不受惩罚的目标残差截距 $\delta_\alpha$，最终预测为

\[
X\left(\widehat w+\widehat\delta\right)
+\widehat w_\alpha+\widehat\delta_\alpha. \tag{TF-I}
\]

这是为合成生成式均值偏移设置的公平扩展，不是原文式（2）（5）的严格复现，也不继承原定理。它必须与 `paper_exact` 分别使用完整的共同 12 点网格、单独命名并报告全部结果；不得在两个版本之间按 validation 择优后只报告一个，也不得替代 `paper_exact`。

把常数列直接拼进 $X$ 也不等于“不惩罚截距”：按原式，该列对应的 $\beta^{(0)}$ 会受到 $\lambda_0$ 惩罚，各源—目标截距差会受到 $\lambda_0a_k$ 惩罚。除非明确登记为第三种偏离版本，否则禁止这样实现。

## 6. 数值求解证书

两个问题都是凸的，但“求解器返回数组”不足以证明复现正确。所有 12 个 $c$、两个 $n_0$ 组都要用 float64 求解，并从原始未拆分公式重新计算目标值。

当前环境已有 SciPy 和 scikit-learn，但它们不因此自动构成式（TF-2）的 fused-ℓ1 求解器；本理论分支不安装依赖。独立实验分支可通过 `uv` 新增并锁定 cvxpy 及其实际使用的凸求解器版本，提交锁文件并在结果元数据中记录版本。无论采用何种库，下述原始目标复算和 KKT 容差才是数值验收标准，不能以库名代替证书。

对第一步，定义

\[
g_k=\frac{1}{N}(X^{(k)})^\top
(X^{(k)}\widehat\beta^{(k)}-y^{(k)}),
\qquad d_k=\widehat\beta^{(k)}-\widehat\beta^{(0)}. \tag{K-1}
\]

必须能从求解器对偶量或显式子梯度恢复 $u_0\in\partial\|\widehat\beta^{(0)}\|_1$、$u_k\in\partial\|d_k\|_1$，并验证

\[
g_k+\lambda_0a_ku_k=0\quad(k\geq1),
\qquad
g_0+\lambda_0u_0-\lambda_0\sum_{k=1}^{K}a_ku_k=0. \tag{K-2}
\]

非零坐标的子梯度取符号，数值零坐标必须落在 $[-1,1]$。对第二步，定义

\[
g_\delta=\frac{1}{n_T}(X^{(0)})^\top
\left[X^{(0)}(\widehat w+\widehat\delta)-y^{(0)}\right], \tag{K-3}
\]

并验证存在 $v\in\partial\|\widehat\delta\|_1$ 使

\[
g_\delta+\widetilde\lambda v=0. \tag{K-4}
\]

自检程序固定计算

\[
R_1=\max\left\{
\max_{k\geq1}\|g_k+\lambda_0a_ku_k\|_\infty,
\left\|g_0+\lambda_0u_0-\lambda_0\sum_{k=1}^{K}a_ku_k\right\|_\infty
\right\}, \tag{K-5}
\]

\[
D_1=\max\left(1,\max_k\|g_k\|_\infty,
\lambda_0,\max_k\lambda_0a_k\right),
\qquad
R_2=\|g_\delta+\widetilde\lambda v\|_\infty,
\qquad
D_2=\max(1,\|g_\delta\|_\infty,\widetilde\lambda). \tag{K-6}
\]

每次求解的强制证书为：求解器状态为 `optimal`，不能接受 `optimal_inaccurate`；若采用拆分变量，其最大原始约束违反不超过 $10^{-8}$；$\max(R_1/D_1,R_2/D_2)\leq10^{-6}$；重新计算的原始目标有限，且与求解器报告值的相对差不超过 $10^{-8}$。截距公平扩展还须验证每个第一步非惩罚截距的损失梯度为零，

\[
\frac{1}{N}\mathbf 1^\top
\left(X^{(k)}\widehat\beta^{(k)}
+\widehat\alpha_k\mathbf 1-y^{(k)}\right)=0,
\tag{K-I1}
\]

以及第二步非惩罚残差截距的损失梯度为零，

\[
\frac{1}{n_T}\mathbf 1^\top
\left[X^{(0)}(\widehat w+\widehat\delta)
+(\widehat w_\alpha+\widehat\delta_\alpha)\mathbf 1-y^{(0)}\right]=0,
\tag{K-I2}
\]

两者均采用相同的 $10^{-6}$ 相对 KKT 容差。任一阶段不满足时，允许用更严容差和更多迭代原配置重算一次；仍失败则该固定 $c$ 所在的整次数据格—种子—TransFusion 版本标记为 `ENGINEERING_FAILURE`，不得删去该 $c$ 后用缩小的 validation 网格继续排名，也不得换目标缩放、截距口径或正则映射挽救。

每次拟合至少记录：数据组、随机种子、$c$、$n_0$、$n_S$、$N$、$\lambda_0$、$a_k$、$\widetilde\lambda$、两阶段原始目标值、求解器及版本、状态、迭代数、原始/对偶残差、KKT 无穷范数残差、validation MSE、最终系数和预测的 SHA-256。相同输入与配置重复运行时，validation/test 预测最大绝对差须小于 $10^{-8}$。

## 7. 许可与复现来源

论文 PDF 首页标明“Copyright 2024 by the author(s)”。PMLR 的官方 [Publication Agreement](https://proceedings.mlr.press/pmlr-license-agreement.pdf) 第 2 条将文章以 Creative Commons Attribution 4.0 International（CC BY 4.0）许可给公众，第 3 条要求归属声明包含原 proceedings 引用及指向 PMLR 原论文的链接，第 4 条说明作者保留版权。因此实验分支和报告必须引用原论文并保留上述 PMLR 链接。

CC BY 4.0 是文章许可，不是软件许可证。截至 2026-09-09，官方论文页只提供论文 PDF，未列出作者代码仓库或代码许可证。本项目只能根据式（2）（5）独立实现，不复制来源和许可证不明的第三方实现。所用数值求解库必须在实验分支另记名称、版本和软件许可证；论文许可不能替代依赖库许可。

## 8. 工程交接清单

- 实现中央式 `paper_exact`，严格使用 $1/(2N)$ 和 $1/(2n_0)$ 两种残差缩放。
- 输出式（TF-w），再做式（TF-5a）的目标残差 Lasso；不得跳步。
- 两个目标标签组固定为 $n_0=96$ 和 $n_0=768$，分别计算 $N$、聚合权重和正则系数。
- 只运行共同 12 点 $c$ 网格；不拆分搜索 $c_0,c_1$，不在理论事件分支间择优。
- 保留原式无截距主结果；强制公平非惩罚截距扩展单独命名、完整运行且不得替代主结果。
- 每个候选必须通过两阶段 KKT 和原始目标复算证书后才能参与 validation 选择。
- 不读取真实站点数据，不训练 PatchTST，不修改 TACBR 或原 F 组放行门。

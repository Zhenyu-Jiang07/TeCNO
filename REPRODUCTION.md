# TeCNO + Cholec80 复现记录

项目：`/data/users/jiangzhenyu/projects/tecno-reproduction`。日期：2026-10-07（Asia/Shanghai）。
原始提交：`0c4313f3765348f5e42a8b7e9183639aea0bca4e`。

## 状态

- 已核对论文、原始实现及作者 issues；独立 `tecno` 环境已创建，依赖检查通过。
- RTX 5090 上 ResNet50 和 2×9 层 TeCNO 前向、反向通过；CNN batch=64 实测峰值约 5.5 GiB。
- 完整 TCN 的 GPU 因果性测试也通过：未来扰动对应的历史输出最大误差 0，未来梯度非零项 0，截断前缀误差约 4.2e-7；见 `reports/gpu-causality.json`。
- 正式加载配置（真实训练病例 36，batch 64，workers 8）连续 18 步更新通过，约 686 帧/秒，峰值 5.5 GiB；最长 29,966 时间点的合成 TCN 资源测试也通过，峰值约 0.6 GiB。两者都不作为训练成绩；见 `reports/real-loader-gpu-check.json`、`reports/full-length-tcn-gpu-check.json`。
- 官方公开数据正在分块下载；80 个病例的阶段、器械标注已取得并核对零起始连续帧号。
- 5 项关键单元测试通过（包含因果性、未来梯度、非因果对照、指标和 logits CE）。
- 真实病例小流程已通过：36 号训练、41 号验证；CNN 和 TCN 各跨两个独立进程保存/恢复模型及优化器；2048 维特征缓存复用通过；验证集评测和时间轴生成通过。证据：`runs/smoke-real/completed.json`。这些是极小子集测试，分数不作为实验成绩。
- 2026-10-07 02:21（Asia/Shanghai）：01–48 全部准备完成、训练统计完成，正式 CNN 训练开始。下载/校验/解压/测试病例预处理同时继续。**训练仍在进行，尚无测试集复现成绩。**
- 运行状态以 `logs/`、`runs/` 的 JSON/JSONL 记录为准；不得把下载、冒烟测试或训练启动称为复现完成。

## 依据与实验协议

来源：[论文 arXiv v1](https://arxiv.org/abs/2003.10751)、[作者代码](https://github.com/tobiascz/TeCNO/tree/0c4313f3765348f5e42a8b7e9183639aea0bca4e)、[架构参数的作者答复](https://github.com/tobiascz/TeCNO/issues/6#issuecomment-802851853)、[器械标签扩展的作者答复](https://github.com/tobiascz/TeCNO/issues/5#issuecomment-802849431)。本地原文和答复在 `sources/`。

| 项目 | 论文/作者说明 | 原仓库默认 | 本次采用 |
|---|---|---|---|
| 病例划分 | 40 train / 8 val / 32 test | 01–40 / 41–48 / 49–80 | 同代码病例编号，两个训练环节一致 |
| 采样率 | 5 fps；文中约 92,000 帧与标注数量不符 | CNN 1 fps；TCN 假定输入 25 fps 再降至 1 fps | 每视频原帧号 0,5,10,…；标注合计 922,570 个 5 fps 帧，训练 431,560；最终数以视频解码审计为准 |
| 视觉编码器 | ResNet50；阶段+器械多任务 | ImageNet 预训练，两头各 7 类 | 原模型，明确使用 torchvision ImageNet V1 权重，2048 维池化特征 |
| 视觉损失 | 加权阶段 CE + 器械 BCE | 学习两个 log variance 进行自动任务平衡 | 保留代码自动平衡；总损失可以为负，不等于数值错误 |
| CNN 训练 | 独立的轮数、batch 和增广细节未充分说明 | Adam 5e-4；最多 15 epochs；batch 写 1000 但被限制为 80；早停 patience=3 | Adam 5e-4，15 epochs，batch 64，固定轮数，验证逐视频准确率选优；不早停 |
| TCN 结构 | 2 stages；作者确认 9 layers / 64 maps / 2048 input | 配置 8 layers / 32 maps | 作者确认的 2 / 9 / 64 / 2048，kernel=3，dilation=1…256，dropout=.5，causal |
| TCN 损失 | 各 stage 加权 CE 的平均 | softmax 后再次调用 CE | 修正为 logits→CE；仍取两 stage 平均；无额外平滑损失 |
| TCN 训练 | Adam 5e-4，25 epochs，整段视频，验证集选优 | 7e-5，12–16 epochs，早停 | 5e-4，25 epochs，每 batch 一个完整视频；验证逐视频准确率选优 |
| 类别权重、归一化 | median frequency balancing；归一化细节未说明 | 写死权重和 RGB 统计值，计算范围不可核实 | 权重只用训练帧：median(count)/count；RGB 统计只用训练病例确定性的 1 fps 子样本 |
| 输入预处理 | 未充分说明 | 视频先缩放 250×250 PNG，再缩放 224×224；ShiftScaleRotate | 保持同尺寸和增广参数，保存 5 fps PNG；Albumentations 1.3.1 的等价 API，记录版本差异 |
| 重复次数 | 5 次随机初始化的均值±标准差 | 未固定种子 | 先完成 seed=2334 单次，再决定是否重复；不能称为复现五次均值 |

原始视频 25 fps，阶段标注 25 fps，器械标注 1 fps。器械标签使用最近一次标注向后保持；官方结尾比最后器械标注多 25 帧，15/37 号阶段标注额外多 1 帧，5 fps 采样不取多出的那一帧。保留末端 age=25 的一帧，逐视频记录，不复制原代码写死的两个全局删除索引。`frame` 永远是原视频零起始帧号；不使用跨病例 DataFrame `iloc[::factor]`。

编码器只能在 01–40 训练、41–48 选 checkpoint。冻结最佳编码器后提取全部病例特征；提取时不更新 BatchNorm，不随机增广，不计算测试表现。TCN 只加载 train/val 特征；最终模型固定后才评测 49–80。测试阶段不选择 stage/checkpoint、阈值或平滑参数。

权重审计：训练病例 5 fps 标注统计得到约 `[1.922, 0.196, 0.985, 0.299, 1.943, 1.000, 2.197]`，与仓库常量不同（Preparation、GallbladderDissection 分别相差约 17%、20%）。仓库常量也不完全等于当前官方数据的 01–48 或 01–80 全帧统计，因此不能断言其来源或断言原作者使用了测试分布。本次只采用解码后训练 manifest 计算出的权重；对其他范围的统计仅用于查明原常量来源，不进入训练。证据：`reports/training-weight-audit.json`、`reports/legacy-weight-origin-audit.json`。

## 评测与比较边界

最后一个 TCN stage 的 argmax；严格 5 fps 帧评测，不提供边界容忍区间，不使用未来帧或离线后处理。保存每视频混淆矩阵、逐帧预测 CSV、宏观指标和首个测试视频时间轴。

- 主准确率：先算每视频正确帧比例，再对视频平均；另报全帧池化准确率，避免长视频权重歧义。
- Precision/Recall：每视频、每阶段计算；该分母为零记 undefined，从该阶段跨视频平均中排除，然后七阶段平均；同时报告各阶段有效视频数。
- 原实现的累计 metric 在 batch 中 compute 后再次平均会依赖遍历顺序；新实现直接从独立视频混淆矩阵聚合。
- 论文对跨视频聚合及未出现阶段的处理说明不足，因此不能声称完全等价的官方评测。文中对应 ResNet50、双阶段、多任务监督设置：Acc **88.56±0.27%**，Prec **81.64±0.41%**，Rec **85.24±1.06%**。这些是五次运行的统计；本次单种子的跨视频标准差另列，不能混用。
- 单 stage 感受野 1023 个采样点；串联两 stage 最终感受野 2045 个采样点（5 fps 约 409 秒）。只使用过去和当前；单元测试检查未来扰动、前缀一致性、未来输入梯度为零，并加入非因果对照。

## 环境

独立前缀 `/data/users/jiangzhenyu/miniconda3/envs/tecno`；没有修改 base、其他环境或 GPU 驱动。
Python 3.11，PyTorch 2.7.1+cu128，torchvision 0.22.1+cu128，Lightning 2.5.2。
选择依据：[PyTorch 2.7 官方 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。CUDA 运行库随环境 wheel 安装，`nvidia-smi` 的 CUDA 13.2 表示驱动能力，不是必须安装的 PyTorch 版本。

首次创建命令在 `scripts/setup_env.sh`。本次实际依次执行：

```bash
/data/users/jiangzhenyu/miniconda3/bin/conda create --name tecno python=3.11 pip --override-channels -c conda-forge -y
/data/users/jiangzhenyu/miniconda3/envs/tecno/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
/data/users/jiangzhenyu/miniconda3/envs/tecno/bin/python -m pip install -r requirements-reproduction.txt
/data/users/jiangzhenyu/miniconda3/bin/conda install --name tecno ffmpeg --override-channels -c conda-forge -y
```

精确版本：`reports/environment-lock.yml`、`reports/conda-explicit.txt`、`reports/pip-freeze.txt`。
GPU 验证：`reports/gpu-validation.json`；驱动：`reports/nvidia-smi.txt`；安装日志：`logs/setup/`。

## 数据与目录

[CAMMA 官方入口](https://camma.unistra.fr/datasets/)，[官方压缩包](https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip)。无需账号或交互申请。许可 CC-BY-NC-SA 4.0；应同时引用 EndoNet/Twinanda 等人及 TeCNO。原包 README 保存在 `data/raw/cholec80/README.txt`。

- 远端大小 74,916,858,781 bytes（69.8 GiB），324 entries，解压总量约 70.0 GiB；Last-Modified 2022-09-13；ETag、来源、版本在 `data/raw/cholec80.chunks/source.json`。
- 下载前 `/data` 可用约 804 GiB，足够压缩包、原视频、5 fps PNG、特征和 checkpoints。
- `data/raw/cholec80.zip` 下载中是稀疏文件，不能按表面文件长度判断完成；每 32 MiB 分块有 SHA256 marker，完成后另写全包 SHA256。
- 解压时逐文件 ZIP CRC32 和 SHA256；`*.verified.json` 保存记录。
- `data/processed/cholec80_5fps/frames/`：250×250 PNG；`manifests/`：原帧号与标签；`audit/`：逐视频解码对齐审计；`training_statistics.json`：训练统计。
- `features/cholec80_multitask_5fps_seed2334/`：每视频浮点 2048 维特征、原帧号、标签、CNN logits；provenance 绑定编码器 checkpoint SHA256，匹配时重用缓存。
- `runs/cholec80_multitask_5fps_seed2334/{cnn,tcn}/`：完整配置、每 epoch 记录、CSV/TensorBoard、best 和 last checkpoints；`evaluation/test/`：最终结果、逐帧 CSV、PNG/PDF 时间轴。
- 所有数据、特征、checkpoint、日志、下载资料均已加入 `.gitignore`。

## 代码改动与可重跑命令

原 `models/mstcn.py` 保持不变；`models/cnn.py` 只把旧 `pretrained=True` 显式映射到 ImageNet V1。原 Lightning 0.8.5 requirements 与代码本身也不一致，且不支持本 GPU 运行栈；新增 `reproduction/` 将旧 trainer、数据准备、metric hook 适配到 Lightning 2，继续直接导入作者的 CNN/TCN 模型。保留原入口和模块作对照，新流程不用 WandB 账号。TCN 交叉熵把 batch×time 展平为样本维，数值定义不变，规避该 CUDA 版本 `nll_loss2d` 不支持严格确定性的限制；不降低全局确定性要求。

```bash
cd /data/users/jiangzhenyu/projects/tecno-reproduction
export PATH=/data/users/jiangzhenyu/miniconda3/envs/tecno/bin:$PATH
python scripts/download_cholec80.py
python -m reproduction.prepare --unpack --statistics
python -m pytest -q tests/test_reproduction.py
python -m reproduction.train cnn
python -m reproduction.extract
python -m reproduction.train tcn
python -m reproduction.evaluate --split test
```

实际后台入口：`bash scripts/start_pipeline.sh`。状态：`logs/pipeline-status.json`，总日志 `logs/pipeline.log`，阶段日志 `logs/formal-{statistics,cnn,features,tcn,evaluation}.log`。可重复运行入口：正在运行时不会另开第二个流程，重启后接管仍存活的本项目子进程，并检查已有完成标志和 last checkpoint。监督器每 10 秒检查，约每分钟写一次总日志。下载及解压可恢复；普通阶段失败最多重试三次，持续失败会保存 `failed` 和错误位置，不假报完成。外部 GPU 占用时等待。只有真实小流程通过、01–48 全部预处理完成后才允许正式 CNN 训练；全包 SHA256、全部病例准备和 CNN 完成后才提取正式特征；测试评测排在两次训练和全特征缓存完成之后。下载最多 32 条 HTTP range 连接，预处理最多 3 个视频进程；不调整系统服务或其他人的作业。

首次小流程曾发现推理配置目录与 smoke checkpoint 目录不一致、时序 CE 的确定性 CUDA 算子限制；已修正并从已保存的 CNN checkpoint 接续，没有重新使用测试集或改变训练协议。历史错误保留在 `logs/smoke-real.log`，最终通过标志单独保存。

训练默认从同一运行目录的 `last.ckpt` 恢复，保存模型、优化器、epoch/global step 和 RNG 状态；恢复边界为已完成的 epoch。改变配置必须换运行目录，不能静默混入旧实验。每次 GPU 作业前检查 compute 进程；发现其他作业则等待，不终止它们。种子、数值精度为 32-bit、TF32 关闭、确定性 cuDNN 设置均有记录。

## 待完成

1. 完成全数据下载、逐病例预处理及训练统计；核对视频帧数和标注尾部。
2. 已通过真实病例 CNN/TCN 小训练、checkpoint 恢复、特征提取和验证评测。
3. 完成正式 CNN、特征缓存和 TCN 训练，最后评测锁定测试集。
4. 写出实际结果、与论文差距和原因，并展示真实测试时间轴。

# TeCNO 单次复现实验结果

完成时间：2026-10-07T05:26:11.141080+08:00。种子：2334。训练 01–40，验证 41–48，测试 49–80；5 fps；CNN 阶段+器械监督。

CNN 完成 15 轮、101,160 步，选中第四轮（验证 Acc 83.92%）；TCN 完成 25 轮、1,000 次整视频更新，选中第六轮（验证 Acc 90.76%）。正式阶段无失败重试，GPU 计算资源已释放。05:27（Asia/Shanghai）独立核验通过：全部 80 个特征文件、32 测试视频的原始标签及 sklearn 重算指标均一致。

论文参考为 [TeCNO arXiv v1 中 Cholec80、ResNet50、双阶段、多任务视觉编码器的对应设置](https://arxiv.org/abs/2003.10751)。本次仅一个种子；全帧池化 Accuracy 与逐视频平均是不同指标，不能混用。

| 指标 | 本次 TeCNO | 本次 CNN | 论文（五次均值±标准差） |
|---|---:|---:|---:|
| 逐视频平均 Acc | 87.62% | 80.58% | 88.56±0.27% |
| 全帧 Acc（补充） | 86.10% | 79.07% | 聚合方式未充分说明 |
| 宏 Precision | 82.23% | 69.37% | 81.64±0.41% |
| 宏 Recall | 84.86% | 76.39% | 85.24±1.06% |

与论文 Acc 均值的差为 -0.94 个百分点。本次是完整单种子训练和严格帧评测，不是五次结果的统计复现。
论文评测聚合细节不完整；CNN 训练细节、现代依赖、训练统计和损失输入修正均见 REPRODUCTION.md，不能仅凭分数宣称方法完全等价或达到论文结果。
**状态区分：代码跑通 ✓；完成训练和测试 ✓；全面达到论文结果：尚不能宣称。** 当前 Accuracy 比论文参考均值低 0.94 个百分点，Precision 高 0.59 个百分点，Recall 低 0.38 个百分点。本次没有使用测试集选 checkpoint 或调参。

## 观察与误差分析

- 时序模型相对本次同一编码器的 CNN 基线，Accuracy 提高 **7.04** 个百分点；32 个测试病例中 31 个提高，76 号略降约 0.32 个百分点。
- 相邻采样帧的预测阶段切换总数由 CNN 的 **30,983** 降到 TeCNO 的 **3,246**，减少约 **89.52%**；真实标注为 187 次。短时跳变明显减少，但仍有过分割和错误的连续阶段。这是描述性分析，不作为新增的论文对比指标。
- 困难类别包括 CleaningCoagulation（Precision 64.27%、Recall 73.69%）和 Preparation（Recall 72.60%）。全测试集最多的混淆是 CalotTriangleDissection 被识别为 GallbladderDissection（17,443 个采样帧）。
- 最长的 58 号视频有 29,966 个采样帧，CNN/TeCNO Accuracy 为 **54.26%/59.19%**；CalotTriangleDissection→GallbladderDissection、ClippingCutting→GallbladderDissection 错误较多。长而困难的视频也解释了全帧池化 Acc 86.10% 低于逐视频平均 87.62%。
- 论文 No-TCN ResNet50 Accuracy 参考为 82.22%，本次 CNN 为 80.58%。在上述可比性限制下，视觉编码器是差距的一个候选来源。CNN 轮数、增广/归一化、batch 64 与仓库有效 batch 80、训练集重新计算的类别权重、现代依赖以及 TCN 重复 softmax 修正都已记录；尚无受控实验能将差距归因于某一项。
- CNN 验证最优在第 4 轮，TCN 最优在第 6 轮，均按预先固定的验证准确率选取。后续轮次没有稳定超过最优点，因此最终使用最佳 checkpoint。

当前已获得可追溯的单次基线。用于论文统计对比时，应先固定剩余协议约定，再按预先确定的种子重复实验并报告均值/标准差。本次测试结果只用于报告和分析，没有据此修改超参数或重新选模。

## 真实测试时间轴与验证曲线

49 号为按编号选择的首个测试视频，TeCNO Accuracy **93.81%**，CNN 为 85.04%。图覆盖整段视频；TeCNO 预测更连续，仍存在边界提前/延迟和局部误判。

![video49: ground truth, CNN, TeCNO](/data/users/jiangzhenyu/projects/tecno-reproduction/runs/cholec80_multitask_5fps_seed2334/evaluation/test/video49-timeline.png)

验证曲线仅使用 41–48 号病例，红点标记最终选中的模型：

![validation curves](/data/users/jiangzhenyu/projects/tecno-reproduction/reports/validation-curves.png)

## 审计及复查

独立核验：`reports/verified-results.json`；逐视频和逐阶段统计：`reports/test-per-video.csv`、`reports/test-per-phase.csv`。环境、原始数据校验、命令和代码修改见 `REPRODUCTION.md`。

```bash
cd /data/users/jiangzhenyu/projects/tecno-reproduction
/data/users/jiangzhenyu/miniconda3/envs/tecno/bin/python -m reproduction.audit_results
```

机器可读结果：`/data/users/jiangzhenyu/projects/tecno-reproduction/runs/cholec80_multitask_5fps_seed2334/evaluation/test/results.json`。
测试视频 49 的真实/预测时间轴：`/data/users/jiangzhenyu/projects/tecno-reproduction/runs/cholec80_multitask_5fps_seed2334/evaluation/test/video49-timeline.png`。
每视频预测和混淆矩阵保存在同一评测目录；训练配置、日志及 checkpoint 分别位于 `/data/users/jiangzhenyu/projects/tecno-reproduction/runs/cholec80_multitask_5fps_seed2334/cnn` 和 `/data/users/jiangzhenyu/projects/tecno-reproduction/runs/cholec80_multitask_5fps_seed2334/tcn`。

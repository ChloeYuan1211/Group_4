# DINO Vision Branch Fine-tuning: Experimental Setup

本文档以论文实验部分的写法整理 `move_can_pot` 任务中的 DINO 视觉分支微调方案。这里不展开 DINO feature 数据集的构造过程，而是从已经得到的 image-feature paired dataset 开始，说明训练目标、数据划分、采样策略、模型设计、优化设置、消融尝试和最终 checkpoint 选择逻辑。

最终采用的模型为 **v2 last-4 fine-tuning 的 2000 step checkpoint**。选择依据不是最低离线蒸馏 loss，而是 RoboTwin 下游闭环评估中的实际任务表现。

## 1. Task and Objective

本实验基于已经完成 OpenVLA-OFT 单任务训练的 `move_can_pot` baseline。baseline 已经训练到 100k step，我们不重新训练完整 VLA 模型，而是只对其中的 DINO vision branch 做后续适配。

实验目标可以概括为：

- 只更新 OpenVLA-OFT 视觉骨干中的 DINO 分支。
- 保持 language model、action head、proprio projector 和 tokenizer 不变。
- 保持 DINO 输入输出接口不变，使微调后的视觉分支可以无缝接回原 OpenVLA-OFT 推理流程。
- 通过 DINO feature distillation 改善视觉表征，但最终以 RoboTwin 下游闭环成功率作为模型选择标准。

这个目标决定了本实验不是一个单纯的 feature reconstruction 任务。离线 feature loss 只反映 student DINO 与 teacher target 的接近程度，而实际 robotic policy 还依赖视觉特征与原 action head 之间的兼容性。因此我们在训练阶段记录离线指标，在模型选择阶段使用下游评估做最终判断。

## 2. Dataset and Split

训练使用已经处理好的 image-feature paired dataset。每个样本包含当前帧图像、对应的 teacher DINO patch feature，以及 patch-level 的 `valid_count` 置信度信息。

数据包含两个 domain：

| Domain | Episodes | Role |
| --- | ---: | --- |
| randomized | 500 | 主训练分布，包含更多随机化场景，用于提升泛化和鲁棒性 |
| clean | 50 | 更干净、更稳定的任务分布，用于补充高质量监督信号 |

数据按 episode 级别切分，而不是按 frame 随机切分：

| Domain | Train episodes | Validation episodes | Split ratio |
| --- | ---: | ---: | ---: |
| randomized | 450 | 50 | 90% / 10% |
| clean | 40 | 10 | 80% / 20% |

采用 episode-level split 的原因是避免 temporal leakage。如果同一条 episode 中相邻 frame 同时出现在训练集和验证集中，验证集 loss 会被相邻时间步的强相关性低估，不能真实反映 held-out episode 上的泛化能力。因此，训练和验证严格使用不同 episodes。

本文中的“测试”分为两层：

1. **Offline validation**：在 held-out episodes 上计算 DINO feature distillation loss，用于观察训练趋势和筛选候选 checkpoint。
2. **Downstream closed-loop evaluation**：将微调后的 DINO branch 接回 OpenVLA-OFT，在 RoboTwin 环境中执行完整 policy rollouts，用于最终模型选择。

## 3. Temporal Sampling Strategy

每个 episode 是一段连续时间序列。如果直接使用全部 frames，训练成本较高，而且相邻 frames 信息高度冗余。因此我们使用 stride-based temporal sampling。

具体实现上，先将一个 episode 内的 frames 按时间顺序排列成序列：

```text
F = [f0, f1, f2, ..., fT]
```

给定 stride `s`，训练时每个 epoch 会为该 episode 随机选择一个起始位置 `r in {0, ..., s-1}`，然后在这个有序序列上等间隔取样：

```text
F[r], F[r+s], F[r+2s], ...
```

这种做法有两个作用：

- 单个 epoch 内减少相邻 frame 的重复监督，控制训练成本。
- 不同 epoch 使用不同起始位置，使模型在多轮训练中逐步看到同一 episode 的不同时间步，提高总体时间覆盖率。

验证时固定使用起始位置 `r = 0`，保证每次 offline validation 可复现，避免验证指标受随机起始位置影响。

v1 和 v2 的 stride 设置如下：

| Version | Randomized stride | Clean stride | Design intention |
| --- | ---: | ---: | --- |
| v1 DINO-LoRA | 4 | 2 | 较稀疏采样，控制 8 卡训练成本 |
| v2 last-k fine-tuning | 2 | 1 | 提高时间覆盖率，尤其 clean domain 使用全帧监督 |

v2 将 randomized stride 从 4 减小到 2，使 randomized domain 中每个 epoch 看到约两倍的时间步。clean domain 从 stride 2 改成 stride 1，即每个 epoch 使用所有 clean frames。这样设计的原因是 clean 数据只有 50 episodes，数量少但监督质量高；对 clean domain 做全帧使用，可以让模型更充分学习稳定、清晰的任务视觉模式，同时 randomized domain 仍然保留一定子采样以控制冗余。

按该设置估算，每个 epoch 的训练样本覆盖如下：

| Version | Estimated train frames / epoch | Randomized val frames | Clean val frames |
| --- | ---: | ---: | ---: |
| v1 | 20,362 | 1,936 | 765 |
| v2 | 40,724 | 3,846 | 1,525 |

因此，v2 的一个核心改进不是增加新数据，而是在已有数据上提高 temporal coverage，让 DINO branch 在更多关键动作阶段上接受监督。

## 4. Model and Distillation Target

student model 使用 DINOv2 ViT-L 视觉骨干，并从 100k step OpenVLA-OFT baseline 的 vision backbone 初始化。若 baseline 中包含已有 LoRA 权重，则先将 LoRA 合并到普通 DINO 权重中，作为后续微调的初始化。

student 的输出定义为 DINOv2 ViT-L 的 second-to-last block patch tokens：

```text
student_output shape = [B, 256, 1024]
```

这里 `256` 对应 16 x 16 patch grid，`1024` 是每个 patch token 的 feature dimension。选择 second-to-last layer 是为了保持与 OpenVLA-OFT 当前视觉接口一致；最后一个 block 不参与当前 student target，因此在 v2 中不会被训练。

teacher target 是已经处理好的 DINO patch feature。训练目标是让 student patch tokens 对齐 teacher target，同时尽量不破坏 OpenVLA-OFT 原 action head 已经适配过的视觉特征分布。

## 5. V1: DINO-LoRA Baseline

最初尝试的是 v1 DINO-LoRA。该方案采用参数高效微调：冻结 DINO backbone 的原始权重，只在参与输出的 DINO blocks 上插入 LoRA adapter。

v1 的主要设置为：

| Item | Setting |
| --- | --- |
| Fine-tuning method | DINO-LoRA |
| Trainable range | blocks 0-22 |
| Excluded block | block 23, because it is not used by the student output |
| Target modules | attention QKV, attention projection, MLP fc1, MLP fc2 |
| LoRA rank | 32 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| Trainable parameters | about 12.1M |

v1 使用的 distillation loss 为 cosine loss 加 MSE loss：

```text
L_total = L_cos + L_mse
L_cos = 1 - mean(cosine(student_patch, teacher_patch))
L_mse = mean(sum((student_patch - teacher_patch)^2 over feature dimension))
```

这里的 MSE 在 1024 维 feature 上先求和再平均，因此数值量级很大，离线 validation loss 通常在几百量级。v1 的 offline loss 确实持续下降，说明 LoRA adapter 能逐步拟合 teacher feature。但是下游 RoboTwin 测试效果不理想。

该结果说明：大范围 LoRA 能降低 feature-level loss，但不一定能产生更适合 OpenVLA action head 的 closed-loop 控制表征。也就是说，离线 feature matching 与最终 policy success 并不完全一致。

## 6. V2: Last-k Full Fine-tuning

基于 v1 的问题，v2 将策略从“大范围 LoRA”改为“后层 full fine-tuning”。核心思想是只更新靠近 DINO 输出的最后若干个有效 blocks，而冻结更早层。

这样设计有三个动机：

- DINO 早期层主要编码通用视觉边缘、纹理和局部结构，过多更新可能破坏通用视觉先验。
- 后部 blocks 更直接决定 OpenVLA 使用的高层 patch feature，适合做任务相关适配。
- full fine-tuning 去掉 LoRA 的低秩瓶颈，让有限数量的后层 blocks 有更充分的表达能力。

v2 实际比较了两个 full fine-tuning 方案：

| Variant | Trainable blocks | Additional trainable module | Trainable parameters | Purpose |
| --- | --- | --- | ---: | --- |
| last-4 / full_last4 | 19, 20, 21, 22 | final norm | 50.395M | 主方案，控制更新幅度，保持与 action head 的兼容性 |
| last-6 / full_last6 | 17, 18, 19, 20, 21, 22 | final norm | 75.592M | 容量增强对照，验证更多后层参数是否有帮助 |

在两个方案中，patch embedding、blocks 0 到未解冻 block 之前的所有早期层，以及不参与 student 输出的最后一个 block 都保持冻结。

需要注意的是，teacher target 和 student output 都来自 pre-norm patch tokens，因此 final norm 对当前蒸馏 loss 的直接影响有限。保留 final norm 为 trainable 主要是为了导出兼容和后续实验灵活性；v2 的主要有效更新来自后部 DINO blocks。

## 7. Loss Design

v2 对 loss 做了两个关键修改：balanced patch loss 和 `valid_count` confidence weighting。

首先，v2 将 v1 中 feature dimension 上求和的 MSE 改为 feature dimension 上求平均：

```text
patch_cos = 1 - cosine(student_patch, teacher_patch)
patch_mse = mean((student_patch - teacher_patch)^2 over feature dimension)
```

然后对所有 patches 做平均：

```text
L_total = L_cos + L_mse
```

这样可以让 cosine loss 和 MSE loss 的量级更平衡，避免 1024 维求和 MSE 主导整个优化过程。v2 的离线 loss 因此在 0.x 量级，与 v1 的几百量级不可直接数值比较。

其次，v2 使用 `valid_count` 作为 patch-level 置信度权重：

```text
patch_weight = valid_count / 5.0
```

`valid_count` 表示一个 patch 的 teacher target 由多少个有效来源参与形成。valid_count 越高，说明该 patch target 的可靠性越高；valid_count 越低，说明该 patch target 可能更受遮挡、边界或噪声影响。训练时使用归一化加权平均，使高置信度 patch 对 loss 贡献更大。

这个设计不是重新处理数据，而是在训练目标中显式表达不同 patch target 的可靠程度。它会让模型更重视稳定区域，尤其是任务相关物体和中心操作区域；边缘和低置信度 patch 的影响相对降低。

## 8. Optimization and Hardware

v1 主实验使用 8 张 A100，v2 最终实验使用 1 张 A100。训练脚本本身支持单卡或多卡；核心是通过：

```text
effective_batch_size = per_device_batch_size x gradient_accumulation_steps x world_size
```

控制实际 optimizer step 对应的 batch size。

主要训练参数如下：

| Item | v1 DINO-LoRA | v2 last-4 / last-6 |
| --- | ---: | ---: |
| Hardware | 8 x A100 | 1 x A100 |
| World size | 8 | 1 |
| Per-device batch size | 32 | 8 |
| Gradient accumulation | 1 | 16 |
| Effective batch size | 256 | 128 |
| Epochs | 60 | 20 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-4 | 2e-5 |
| Weight decay | 0.01 | 0.05 |
| Gradient clipping | 1.0 | 1.0 |
| Warmup ratio | 0.05 | 0.05 |
| Precision | TF32 | bf16 autocast + TF32 |
| Eval interval | 200 optimizer steps | 200 optimizer steps |
| Save interval | 200 optimizer steps | 200 optimizer steps |

v1 的 trainable parameters 较少，因此使用更大的 effective batch 和更高 learning rate。v2 改为 full fine-tuning 后部 DINO blocks，可训练参数显著增加，因此使用更小的 learning rate，并通过 gradient accumulation 在单张 A100 上维持稳定训练。

## 9. Offline Validation Trends

offline validation 使用 held-out episodes，指标定义为 randomized validation loss 和 clean validation loss 的平均：

```text
selection_metric = 0.5 x (val_randomized_loss + val_clean_loss)
```

该指标用于观察训练趋势和筛选候选 checkpoint，但不作为最终 winner 的唯一标准。

v1 的 offline loss 持续下降：

| Step | Randomized val loss | Clean val loss | Selection metric |
| ---: | ---: | ---: | ---: |
| 1000 | 369.620 | 325.847 | 347.733 |
| 2000 | 354.501 | 296.875 | 325.688 |
| 4000 | 348.574 | 282.503 | 315.538 |
| 4800 | 348.372 | 281.863 | 315.118 |

v2 last-4 的 offline loss 也持续下降：

| Step | Randomized val loss | Clean val loss | Selection metric |
| ---: | ---: | ---: | ---: |
| 1000 | 0.5829 | 0.5207 | 0.5518 |
| 2000 | 0.5672 | 0.4817 | 0.5244 |
| 3000 | 0.5627 | 0.4648 | 0.5138 |
| 4000 | 0.5611 | 0.4556 | 0.5084 |
| 6000 | 0.5603 | 0.4505 | 0.5054 |

v2 last-6 的 offline loss 更低：

| Step | Randomized val loss | Clean val loss | Selection metric |
| ---: | ---: | ---: | ---: |
| 1000 | 0.5103 | 0.4396 | 0.4750 |
| 2000 | 0.4974 | 0.4075 | 0.4524 |
| 3000 | 0.4949 | 0.3935 | 0.4442 |
| 4000 | 0.4933 | 0.3860 | 0.4396 |
| 6000 | 0.4922 | 0.3809 | 0.4366 |

这些结果说明，更大的更新容量和更长训练确实能让 student 更接近 teacher feature。但是这个趋势不能直接等价于下游 policy 更好。

## 10. Model Selection

最终选择遵循两阶段原则：

1. 先根据 offline validation loss 和训练稳定性筛选多个 candidate checkpoints。
2. 再把 candidate vision branch 接回 OpenVLA-OFT，在 RoboTwin 中做 downstream closed-loop evaluation。

最终采用 **last-4 / full_last4 的 2000 step checkpoint**。

这个选择看似不是 offline loss 最低的 checkpoint，但符合本实验目标。原因是：

- 下游 policy 的成功率不仅取决于 DINO feature 是否接近 teacher，还取决于新视觉特征是否仍然匹配原 OpenVLA action head。
- last-6 的 offline loss 更低，但它更新了更多后层 blocks，可能让视觉特征分布偏离 baseline 更明显。
- last-4 的更晚 checkpoints offline loss 更低，但持续训练也可能带来更强的 feature distribution shift。
- 2000 step checkpoint 是一个中早期平衡点：它已经完成有效的 DINO 任务适配，同时仍较好保留 baseline 视觉表征与 action head 的兼容性。
- RoboTwin closed-loop evaluation 最终表明 last-4 2000 step 的实际任务表现最好，因此它被选为最终模型。

因此，本实验的核心结论不是“offline distillation loss 越低越好”，而是：

```text
适度更新 DINO 后部有效 blocks，并通过下游闭环评估选择 checkpoint，比单纯追求最低离线 feature loss 更适合 OpenVLA-OFT 的视觉分支适配。
```

## 11. Summary for Presentation

论文汇报时可以按以下主线讲：

1. 我们固定 OpenVLA-OFT 的语言和动作模块，只微调 DINO vision branch。
2. 数据使用 randomized 和 clean 两个 domain，并按 episode 做 train/validation split，避免 frame-level leakage。
3. v1 使用 DINO-LoRA，离线 loss 能下降，但下游效果不理想，说明 feature loss 与 policy success 不完全一致。
4. v2 改为 last-k full fine-tuning，只更新接近输出的后部 DINO blocks，并提高 temporal sampling coverage。
5. v2 使用 balanced patch loss 和 `valid_count` confidence weighting，使训练目标更稳定、更关注高置信度 patch。
6. 最终通过 RoboTwin downstream closed-loop evaluation 选择 last-4 的 2000 step checkpoint，而不是选择 offline loss 最低的 last-6 或更晚 checkpoint。

最终一句话总结：

```text
我们从 DINO-LoRA 过渡到 last-k full fine-tuning，通过更合理的时间采样、balanced patch loss 和 patch confidence weighting 获得多个候选视觉分支；最终以 RoboTwin 下游闭环表现为准，选择 last-4 的 2000 step checkpoint。
```

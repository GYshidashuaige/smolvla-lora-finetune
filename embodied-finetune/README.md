# SmolVLA 低显存微调实验

在 RTX 4060 Laptop 8GB 上，将 SmolVLA 预训练模型通过 LoRA 适配到 LIBERO 机器人操作数据。当前为实验项目，结果以 logs 中实际运行记录为准。

**已完成：1000 步微调，导出约 3MB LoRA adapter。** 32 个固定留出样本上的平均 flow-matching loss 从 0.541 降至 0.211；该指标不等于机器人任务成功率。详细配置、验证范围和证据见 [RESULTS.md](RESULTS.md)。权重见 [artifacts/adapter](artifacts/adapter)。

## 环境与来源

- Windows / Python 3.11 / CUDA PyTorch 2.7.1 / torchvision 0.22.1
- 上游：https://github.com/huggingface/lerobot ，v0.4.3，commit `0b067df57d21d3a02d6c511f1609172fa39ac29b`
- 基础模型：https://huggingface.co/lerobot/smolvla_base
- 数据：https://huggingface.co/datasets/lerobot/libero
- 低显存实践参考：https://github.com/wycliffeoleti/smolVLA （其报告指标不属于本项目结果）

## 运行

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe download.py
# 先验证下载、真实数据解码、前向/反向和权重保存
.\.venv\Scripts\python.exe train.py --steps 10 --episodes 4
# 小规模微调，输出目录按时间自动生成
.\.venv\Scripts\python.exe train.py --steps 1000 --episodes 40
# 同样种子、数据、步数下的图像增强对照
.\.venv\Scripts\python.exe train.py --steps 1000 --episodes 40 --augment
```

设置 batch=1、num_workers=0、256px 图像、LoRA rank=16，并冻结大部分基础权重，以降低显存需求。参数需要以本机运行结果校验；小样本训练不保证机器人任务成功率。

## 实验与提交

- `logs/*-command.json`：完整训练命令。
- `logs/*-train.log`：真实训练输出与错误。
- `logs/*-status.json`：进程退出码、配置和产物路径。
- `outputs/`：上游保存的 adapter、处理器和训练状态。
- `.venv/`、模型缓存、数据集、上游源码不加入代码提交。

当前脚本只执行离线模仿学习。训练 loss 不能解释成抓取成功率；没有完成闭环仿真评测前，不声明成功率提升。数据按 episode 选取；正式评估时应另留不参与训练的 episodes，使用固定样本和随机种子进行前后对比，并单独报告闭环任务成功率。

本项目新增 Windows 安装与低显存启动脚本、日志留存和增强对照入口。算法与基础模型归原作者；使用或分发模型、数据、依赖时遵循各自许可证及模型卡条款。

## Windows 兼容与验证说明

`train_entry.py` 使用上游训练入口，仅将 `last` 检查点符号链接改为普通目录复制，避免 WinError 1314；上游源码保持原样。`--policy.input_features=null` 使模型输入从 LIBERO 数据推导；空字典会被配置合并保留旧相机字段，不适用于本实验。

`load_vlm_weights=false` 只跳过单独下载 VLM 权重：完整 SmolVLA 权重仍由 `from_pretrained` 从本地基础模型加载。上游 PEFT 检查会因此打印“from scratch”警告，需结合此前的“Loading weights from local directory”和完整权重 SHA-256 校验理解。

```powershell
# 训练结束后，使用未参与训练的 episodes 40–43 进行固定噪声的离线对照
.\.venv\Scripts\python.exe evaluate.py outputs/<运行目录>/checkpoints/001000/pretrained_model
# 导出可移植的约 3MB adapter（需另行下载基础模型）
.\.venv\Scripts\python.exe export_adapter.py outputs/<运行目录>/checkpoints/001000/pretrained_model
.\package.ps1
```

离线评估指标为 flow-matching loss，比较同一 adapter 模型启用/禁用 LoRA 时对固定留出样本的损失。这只能验证离线适配情况，不能代表闭环抓取成功率。`artifacts/adapter` 中改写了基础模型和 VLM 的路径，使其指向公开模型，而不是本机绝对路径。

# 卫星网络智能路由系统

基于强化学习的LEO卫星网络智能路由算法实现，包括PPO、MAPPO和Dijkstra算法。

## 项目特点

- 支持多种路由算法：
  - PPO (Proximal Policy Optimization)
  - MAPPO (Multi-Agent PPO)
  - Dijkstra (作为基准算法)
- 完整的卫星网络仿真环境
- 支持GPU加速训练
- 支持多种QoS策略
- 详细的性能指标统计和可视化

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/your-username/satellite-routing.git
cd satellite-routing
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 运行Dijkstra算法

```bash
python main.py --mode dijkstra
```

### 训练PPO算法

```bash
python main.py --mode train --algorithm ppo
```

### 训练MAPPO算法

```bash
python main.py --mode train --algorithm mappo
```

### 评估模型

```bash
python main.py --mode eval --algorithm ppo --model_path results/ppo_时间戳/model_final.pth
```

## 项目结构

```
.
├── algorithms/          # 路由算法实现
│   ├── networks/       # 神经网络模型
│   ├── ppo_algorithm.py
│   ├── mappo_algorithm.py
│   └── dijkstra_algorithm.py
├── config/             # 配置文件
├── evaluation/         # 评估模块
├── experiments/        # 实验运行脚本
├── simulation/         # 仿真环境
├── visualization/      # 可视化工具
├── main.py            # 主程序
└── requirements.txt    # 依赖列表
```

## 配置说明

配置文件位于`config/`目录下：
- `base_config.yaml`: 基础配置
- `algorithm_config.yaml`: 算法相关配置
- `link_config.yaml`: 链路相关配置
- `tle_config.yaml`: TLE星座配置

## 性能指标

系统会自动记录以下性能指标：
- 端到端延迟
- 吞吐量
- 丢包率
- QoS满意度
- 路由成功率

## 注意事项

1. 确保有足够的磁盘空间存储训练日志和模型
2. GPU训练需要安装CUDA和相应的PyTorch版本
3. 大规模训练建议使用GPU加速

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请通过Issue与我们联系。 
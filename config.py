"""卫星网络配置参数"""

# 轨道参数
ORBIT_HEIGHT_LEO = 1500  # LEO轨道高度(km)
ORBIT_HEIGHT_MEO = 8000  # MEO轨道高度(km)
NUM_ORBITS_LEO = 16     # LEO轨道数
SATS_PER_ORBIT_LEO = 16 # 每个LEO轨道的卫星数
NUM_ORBITS_MEO = 2      # MEO轨道数
SATS_PER_ORBIT_MEO = 8  # 每个MEO轨道的卫星数
INCLINATION = 55        # 轨道倾角(度)
EARTH_RADIUS = 6371     # 地球半径(km)

# 网络参数
QUEUE_CAPACITY = 300    # 队列容量(MB)
PACKET_SIZE = 15        # 数据包大小(KB)
DATA_GENERATION_RATE = 1.5  # 数据生成率(Gbps)
BANDWIDTH = 20          # 带宽(MHz)
BASE_LOSS_RATE = 0.05   # 基础丢包率(5%)
SNR_MIN = 10           # 最小信噪比(dB)
SNR_MAX = 30           # 最大信噪比(dB)

# 时间参数
UPDATE_INTERVAL = 100        # 更新间隔(ms)
TIME_STEP = 20              # 时间步长(ms)
NETWORK_UPDATE_INTERVAL = 100  # 网络状态更新间隔(ms)

# 路由参数
MAX_PATH_LENGTH = 15    # 最大路径长度

# 强化学习参数
ALPHA = 0.1            # 学习率
GAMMA = 0.9            # 折扣因子
INITIAL_EPSILON = 0.9   # 初始探索率
MIN_EPSILON = 0.1      # 最小探索率
DECAY_RATE = 0.0001    # 探索率衰减速率

# DQN训练参数
NUM_EPISODES = 1000    # 训练回合数
BATCH_SIZE = 32        # 批次大小
TARGET_UPDATE = 10     # 目标网络更新频率
LEARNING_RATE = 0.001  # 学习率
MEMORY_SIZE = 2000     # 经验回放缓冲区大小 
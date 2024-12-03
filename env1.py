# env.py

import numpy as np
import networkx as nx

# 常数定义
c = 3e8  # 光速 (m/s)
kB = 1.38e-23  # 玻尔兹曼常数
T = 290  # 热噪声温度 (K)
WISL = 1.55e-6  # 光信号波长 (m)
omega_t_max = 10  # 最大传输功率 (W)
R_max = 1e9  # 最大传输速率 (bps)
BUFFER_SIZE = 250 * 1024 * 1024  # 缓冲区大小 (字节)
PACKET_SIZE = 20 * 1024  # 数据包大小 (字节)

class SatelliteEnv:
    def __init__(self, service_type='delay_sensitive', multi_agent=True):
        # 仅保留 LEO 卫星网络参数
        self.n_orbits = 4  # 轨道数量
        self.n_sats_per_orbit = 4  # 每轨道卫星数量
        self.n_leo = self.n_orbits * self.n_sats_per_orbit  # 总 LEO 卫星数量
        self.k_paths = 5  # 可用路径数量
        self.multi_agent = multi_agent  # 是否为多智能体
        
        # 轨道参数保持不变
        self.leo_height = 1500 * 1000  # LEO 轨道高度 (米)
        self.orbit_inclination = 55  # 轨道倾角 (度)
        
        # QoS 相关配置保持不变
        self.buffer_size = BUFFER_SIZE
        self.packet_size = PACKET_SIZE
        
        self.qos_requirements = {
            'delay': 50e-3,       # 时延要求 (秒)
            'packet_loss': 0.05,  # 丢包率要求
            'delivery': 0.95      # 交付率要求
        }
        
        # QoS 权重配置保持不变
        self.qos_weights = {
            'delay_sensitive': {'w1': 0.8, 'w2': 0.1, 'w3': 0.1},
            'reliability_sensitive': {'w1': 0.1, 'w2': 0.8, 'w3': 0.1},
            'throughput_sensitive': {'w1': 0.1, 'w2': 0.1, 'w3': 0.8}
        }
        
        self.service_type = service_type
        self.w1 = self.qos_weights[service_type]['w1']
        self.w2 = self.qos_weights[service_type]['w2']
        self.w3 = self.qos_weights[service_type]['w3']

        # 初始化各项状态
        self.leo_positions = self._initialize_positions()
        self.cache_state = np.zeros(self.n_leo)
        self.leo_topology = self._build_leo_topology()
        
        # 区域划分
        self.n_regions = 4
        self.regions = self._initialize_regions()

    def _initialize_positions(self):
        positions = {}
        earth_radius = 6371 * 1000  # 地球半径 (米)
        orbit_radius = earth_radius + self.leo_height

        for orbit in range(self.n_orbits):
            for sat in range(self.n_sats_per_orbit):
                sat_idx = orbit * self.n_sats_per_orbit + sat

                # 计算卫星位置
                theta = 2 * np.pi * sat / self.n_sats_per_orbit  # 在轨道平面内的角度
                phi = np.radians(self.orbit_inclination)  # 轨道倾角
                orbit_phase = 2 * np.pi * orbit / self.n_orbits  # 轨道相位

                # 三维坐标计算
                x = orbit_radius * (np.cos(phi) * np.cos(theta + orbit_phase))
                y = orbit_radius * (np.cos(phi) * np.sin(theta + orbit_phase))
                z = orbit_radius * np.sin(phi)

                positions[sat_idx] = np.array([x, y, z])

        return positions

    def _initialize_coverage(self):
        # 初始化每个 MEO 卫星覆盖的 LEO 卫星
        coverage = []
        satellites_per_meo = self.n_leo // self.n_meo
        for i in range(self.n_meo):
            start = i * satellites_per_meo
            end = (i + 1) * satellites_per_meo
            if i == self.n_meo - 1:
                end = self.n_leo  # 确保覆盖所有 LEO 卫星
            coverage.append(list(range(start, end)))
        return coverage

    def _build_leo_topology(self):
        # 创建 LEO 卫星网络拓扑图（使用完全图模拟，每个卫星与所有其他卫星相连）
        G = nx.complete_graph(self.n_leo)
        return G

    def _initialize_regions(self):
        """基于地理位置将LEO卫星划分为不同区域"""
        regions = []
        # 根据经纬度划分区域
        for i in range(self.n_regions):
            lon_start = -180 + i * 90  # 按经度划分
            lon_end = -180 + (i + 1) * 90
            region_sats = []
            
            for sat_id, pos in self.leo_positions.items():
                # 将笛卡尔坐标转换为经纬度
                lon = np.arctan2(pos[1], pos[0]) * 180 / np.pi
                if lon_start <= lon < lon_end:
                    region_sats.append(sat_id)
            regions.append(region_sats)
        return regions

    def reset(self, src=None, dst=None):
        self.cache_state = np.random.uniform(0, BUFFER_SIZE, size=self.n_leo)
        # 如果未指定源和目的地，则随机选择
        if src is None:
            self.src = np.random.randint(0, self.n_leo)
        else:
            self.src = src
        if dst is None:
            self.dst = np.random.randint(0, self.n_leo)
            # 确保源和目的地不相同
            while self.dst == self.src:
                self.dst = np.random.randint(0, self.n_leo)
        else:
            self.dst = dst
        return self.get_observation()

    def get_observation(self):
        if self.multi_agent:
            # 基于区域返回观察
            return self.get_region_observation()
        else:
            # 返回单智能体的全局观察
            return self.cache_state

    def get_candidate_paths(self, src, dst):
        # 使用 LCSS 算法获取候选路径（这里列举所有简单路径并选取最长的 k 条）
        all_simple_paths = list(nx.all_simple_paths(self.leo_topology, source=src, target=dst, cutoff=4))
        # 排序并选取前 k ���路径
        candidate_paths = sorted(all_simple_paths, key=lambda x: len(x), reverse=True)[:self.k_paths]
        return candidate_paths

    def calculate_distance(self, vi, vj):
        # 计算 LEO 卫星之间的空间距离
        pos_i = self.leo_positions[vi]
        pos_j = self.leo_positions[vj]
        distance = np.linalg.norm(pos_i - pos_j)
        return distance

    def calculate_path_loss(self, ei_j):
        epsilon = 1e-6  # 小常数，防止除以零
        path_loss = 20 * np.log10(4 * np.pi * (ei_j + epsilon) / WISL)
        return path_loss

    def calculate_max_rate(self, path_loss, bandwidth=1e7):
        omega_tr = omega_t_max  # 发射功率 (W)
        G_tr = 1  # 发射天线增益，简化为 1
        G_rc = 1  # 接收天线增益，简化为 1
        noise = kB * T * bandwidth
        snr = (omega_tr * G_tr * G_rc) / (path_loss * noise)
        snr = max(snr, 1e-6)  # 确保 snr 不小于 epsilon

        rate = bandwidth * np.log2(1 + snr)
        rate = min(rate, R_max)
        return rate

    def calculate_qos_metrics(self, path):
        total_delay = 0
        packet_losses = []
        delivery_nums = []
        send_nums = []

        for i in range(len(path) - 1):
            vi = path[i]
            vj = path[i + 1]
            distance = self.calculate_distance(vi, vj)
            propagation_delay = distance / c

            path_loss = self.calculate_path_loss(distance)
            max_rate = self.calculate_max_rate(path_loss)

            mu = max_rate / (self.buffer_size / self.packet_size)
            queuing_delay = 1 / mu
            forwarding_delay = 1 / mu

            total_delay += (propagation_delay + queuing_delay + forwarding_delay)

            buffer_occupancy = self.cache_state[vi] / self.buffer_size
            packet_loss = buffer_occupancy  # 直接关联缓存占用率
            packet_loss = np.clip(packet_loss, 0, 1)
            packet_losses.append(packet_loss)

            eta_n = self.packet_size  # 每步发送固定大小的数据包
            eta_d = eta_n * (1 - packet_loss)
            delivery_nums.append(eta_d)
            send_nums.append(eta_n)

        end_to_end_loss = 1 - np.prod([1 - pl for pl in packet_losses])
        total_eta_d = sum(delivery_nums)
        total_eta_n = sum(send_nums)
        delivery_rate = total_eta_d / total_eta_n if total_eta_n > 0 else 0

        return total_delay, end_to_end_loss, delivery_rate

    def calculate_utility(self, delay, packet_loss, delivery):
        T_hat = min(delay / self.qos_requirements['delay'], 1)
        P_hat = min(packet_loss / self.qos_requirements['packet_loss'], 1)
        D_hat = min(delivery / self.qos_requirements['delivery'], 1)

        U_path = self.w1 * (1 - T_hat) + self.w2 * (1 - P_hat) + self.w3 * D_hat
        return U_path  # 保持为正值，便于最大化

    def update_cache_state(self):
        # 根据传输量确定缓存状态变化，避免使用随机增减
        for i in range(self.n_leo):
            transmitted = min(self.cache_state[i], self.packet_size * self.k_paths)
            received = self.packet_size * 1.5  # 固定接收量
            self.cache_state[i] = np.clip(self.cache_state[i] - transmitted + received, 0, self.buffer_size)

    def step(self, actions):
        if self.multi_agent:
            # 多智能体���理 - 每个区域一个智能体
            total_utility = 0
            candidate_paths = self.get_candidate_paths(self.src, self.dst)
            
            if not candidate_paths:
                done = True
                next_state = self.get_observation()
                return next_state, -1, done
                
            for region_id, action in enumerate(actions):
                if action >= len(candidate_paths):
                    action = 0
                path = candidate_paths[action]
                # 计算基于区域的奖励
                reward = self.calculate_region_reward(region_id, path)
                total_utility += reward
                
            self.update_cache_state()
            next_state = self.get_observation()
            done = False
            return next_state, total_utility, done
        else:
            # 单智能体处理保持不变
            total_utility = 0
            candidate_paths = self.get_candidate_paths(self.src, self.dst)
            # 如果没有可用的路径，结束回合
            if not candidate_paths:
                done = True
                next_state = self.get_observation()
                reward = -1  # 或者给予一个适当的奖励/惩罚
                return next_state, reward, done
            action = actions  # 单智能体的动作
            if action >= len(candidate_paths):
                action = 0  # 防止越界
            path = candidate_paths[action]
            # 计算 QoS 指标
            delay, packet_loss, delivery = self.calculate_qos_metrics(path)
            # 计算效用函数作为奖励（取负数，因为要最小化效用函数）
            utility = -self.calculate_utility(delay, packet_loss, delivery)
            total_utility += utility
            # 更新缓存状态（模拟数据传输影��）
            self.update_cache_state()
            next_state = self.get_observation()
            done = False  # 根据需要设置结束条件
            return next_state, total_utility, done

    def calculate_region_metrics(self, region_id):
        """计算区域内的性能指标"""
        region_sats = self.regions[region_id]
        
        # 1. 区域负载均衡度
        load_balance = 1 - np.std([self.cache_state[sat]/self.buffer_size for sat in region_sats])
        
        # 2. 区域内部连通性
        internal_links = sum(1 for i in region_sats for j in region_sats 
                            if i < j and self.leo_topology.has_edge(i, j))
        max_internal_links = len(region_sats) * (len(region_sats) - 1) / 2
        connectivity = internal_links / max_internal_links if max_internal_links > 0 else 0
        
        # 3. 区域平均缓存使用率
        avg_cache = np.mean([self.cache_state[sat]/self.buffer_size for sat in region_sats])
        
        return load_balance, connectivity, avg_cache

    def calculate_region_reward(self, region_id, path):
        """计算基于区域的奖励"""
        # 基础 QoS 指标
        delay, packet_loss, delivery = self.calculate_qos_metrics(path)
        
        # 区域性能指标
        load_balance, connectivity, avg_cache = self.calculate_region_metrics(region_id)
        
        # 区域路径质量（路径中经过该区域的节点比例）
        region_sats = self.regions[region_id]
        path_quality = sum(1 for node in path if node in region_sats) / len(path)
        
        # 综合奖励计算
        qos_reward = (
            0.4 * (1 - delay/self.qos_requirements['delay']) +
            0.3 * (1 - packet_loss/self.qos_requirements['packet_loss']) +
            0.3 * (delivery/self.qos_requirements['delivery'])
        )
        
        region_reward = (
            0.3 * load_balance +
            0.3 * connectivity +
            0.2 * (1 - avg_cache) +  # 较低的缓存使用率更好
            0.2 * path_quality
        )
        
        # 总奖励为 QoS 奖励和区域奖励的加权和
        total_reward = 0.6 * qos_reward + 0.4 * region_reward
        
        return total_reward

    def get_region_observation(self):
        """返回基于区域的观察"""
        observations = []
        for region_id, region_sats in enumerate(self.regions):
            # 区域内卫星的状态
            cache_states = self.cache_state[region_sats]
            
            # 区域性能指标
            load_balance, connectivity, avg_cache = self.calculate_region_metrics(region_id)
            
            # 区域观察向量
            obs = {
                'cache_states': cache_states,
                'load_balance': load_balance,
                'connectivity': connectivity,
                'avg_cache': avg_cache,
                'src_in_region': self.src in region_sats,
                'dst_in_region': self.dst in region_sats
            }
            observations.append(obs)
            
        return observations
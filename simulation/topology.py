from typing import Dict, Any
import numpy as np

class WalkerConstellation:
    """Walker星座类,实现卫星轨道动力学和链路管理"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Walker星座
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.num_planes = config['orbital_planes']
        self.sats_per_plane = config['satellites_per_plane']
        self.total_satellites = self.num_planes * self.sats_per_plane
        
        # 轨道参数
        self.altitude = config['altitude']  # 轨道高度(km)
        self.inclination = np.radians(config['inclination'])  # 倾角(rad)
        self.phase_diff = 2 * np.pi / self.num_planes  # 轨道面相位差
        self.mean_motion = self._calculate_mean_motion()  # 平均角速度(rad/s)
        
        # 初始化卫星状态
        self.positions = np.zeros((self.total_satellites, 3))
        self.velocities = np.zeros((self.total_satellites, 3))
        self._initialize_constellation()
    
    def _calculate_mean_motion(self) -> float:
        """
        计算卫星平均角速度
        
        Returns:
            float: 平均角速度(rad/s)
        """
        # 地球参数
        G = 6.67430e-11  # 万有引力常数(m^3/kg/s^2)
        M = 5.972e24     # 地球质量(kg)
        R = 6371.0       # 地球半径(km)
        
        # 计算轨道半径(km)
        r = R + self.altitude
        
        # 计算平均角速度(rad/s)
        return np.sqrt(G * M / ((r * 1000) ** 3))
    
    def _initialize_constellation(self) -> None:
        """初始化星座"""
        # 计算每个轨道面的相位偏移
        f = 1  # Walker星座的相位因子
        relative_spacing = f / self.num_planes
        
        for i in range(self.total_satellites):
            plane = i // self.sats_per_plane
            pos_in_plane = i % self.sats_per_plane
            
            # 计算初始相位角
            phase = 2 * np.pi * pos_in_plane / self.sats_per_plane
            # 添加轨道面间的相位偏移
            phase += 2 * np.pi * relative_spacing * plane
            
            raan = self.phase_diff * plane  # 升交点赤经
            
            # 设置初始位置和速度
            self._update_satellite_state(i, phase, raan, 0.0)
    
    def _update_satellite_state(self, 
                              sat_id: int, 
                              phase: float, 
                              raan: float, 
                              time: float) -> None:
        """
        更新卫星状态
        
        Args:
            sat_id: 卫星ID
            phase: 初始相位角(rad)
            raan: 升交点赤经(rad)
            time: 当前时间(s)
        """
        # 计算当前相位角
        current_phase = phase + self.mean_motion * time
        
        # 计算轨道半径(km)
        r = self.altitude + 6371.0
        
        # 1. 在赤道面内的位置
        x = r * np.cos(current_phase)
        y = r * np.sin(current_phase)
        z = 0.0
        
        # 2. 绕x轴旋转（倾角）
        # Rx = [1      0           0     ]
        #      [0   cos(i)   -sin(i)]
        #      [0   sin(i)    cos(i)]
        x_incl = x
        y_incl = y * np.cos(self.inclination) - z * np.sin(self.inclination)
        z_incl = y * np.sin(self.inclination) + z * np.cos(self.inclination)
        
        # 3. 绕z轴旋转（RAAN）
        # Rz = [cos(Ω)   -sin(Ω)   0]
        #      [sin(Ω)    cos(Ω)   0]
        #      [   0         0      1]
        self.positions[sat_id] = np.array([
            x_incl * np.cos(raan) - y_incl * np.sin(raan),
            x_incl * np.sin(raan) + y_incl * np.cos(raan),
            z_incl
        ])
        
        # 计算速度
        v = np.sqrt(398600.4418 / r)  # 轨道速度(km/s)
        
        # 1. 在赤道面内的速度
        vx = -v * np.sin(current_phase)
        vy = v * np.cos(current_phase)
        vz = 0.0
        
        # 2. 绕x轴旋转（倾角）
        vx_incl = vx
        vy_incl = vy * np.cos(self.inclination) - vz * np.sin(self.inclination)
        vz_incl = vy * np.sin(self.inclination) + vz * np.cos(self.inclination)
        
        # 3. 绕z轴旋转（RAAN）
        self.velocities[sat_id] = np.array([
            vx_incl * np.cos(raan) - vy_incl * np.sin(raan),
            vx_incl * np.sin(raan) + vy_incl * np.cos(raan),
            vz_incl
        ])
    
    def update(self, time: float) -> None:
        """
        更新星座状态
        
        Args:
            time: 当前时间(s)
        """
        # 计算每个轨道面的相位偏移
        f = 1  # Walker星座的相位因子
        relative_spacing = f / self.num_planes
        
        for i in range(self.total_satellites):
            plane = i // self.sats_per_plane
            pos_in_plane = i % self.sats_per_plane
            
            # 计算初始相位角
            phase = 2 * np.pi * pos_in_plane / self.sats_per_plane
            # 添加轨道面间的相位偏移
            phase += 2 * np.pi * relative_spacing * plane
            
            raan = self.phase_diff * plane
            
            self._update_satellite_state(i, phase, raan, time)
    
    def get_satellite_position(self, sat_id: int) -> np.ndarray:
        """
        获取卫星位置
        
        Args:
            sat_id: 卫星ID
            
        Returns:
            np.ndarray: 卫星位置(x,y,z)
        """
        return self.positions[sat_id]
    
    def get_satellite_velocity(self, sat_id: int) -> np.ndarray:
        """
        获取卫星速度
        
        Args:
            sat_id: 卫星ID
            
        Returns:
            np.ndarray: 卫星速度(vx,vy,vz)
        """
        return self.velocities[sat_id]
    
    def get_link_data(self, src: int, dst: int) -> Dict[str, float]:
        """
        获取链路数据
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            Dict[str, float]: 链路数据
        """
        # 计算卫星间距离
        distance = np.linalg.norm(self.positions[src] - self.positions[dst])
        
        # 计算链路容量和延迟
        if self._can_establish_link(src, dst):
            # 使用香农公式计算链路容量
            capacity = self._calculate_link_capacity(distance)
            # 计算传播延迟
            delay = distance / 3e5  # 光速传播
            return {
                'capacity': capacity,
                'delay': delay,
                'distance': distance,
                'quality': max(0, 1 - distance / self.config['max_range'])
            }
        else:
            return {
                'capacity': 0.0,
                'delay': float('inf'),
                'distance': distance,
                'quality': 0.0
            }
    
    def _can_establish_link(self, src: int, dst: int) -> bool:
        """
        判断两颗卫星是否可以建立链路
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            bool: 是否可以建立链路
        """
        if src == dst:
            return False
            
        # 获取卫星所在轨道面和位置信息
        plane_src = src // self.sats_per_plane
        plane_dst = dst // self.sats_per_plane
        pos_src = src % self.sats_per_plane
        pos_dst = dst % self.sats_per_plane
        
        # 计算距离
        distance = np.linalg.norm(self.positions[src] - self.positions[dst])
        if distance > self.config['max_range']:
            return False
            
        # 同一轨道面内的相邻卫星
        if plane_src == plane_dst:
            # 考虑环形拓扑
            diff = min(
                abs(pos_src - pos_dst),
                abs(pos_src - pos_dst + self.sats_per_plane),
                abs(pos_src - pos_dst - self.sats_per_plane)
            )
            if diff == 1:
                return True
            
        # 相邻轨道面的卫星(除了极地区域)
        plane_diff = min(
            abs(plane_src - plane_dst),
            abs(plane_src - plane_dst + self.num_planes),
            abs(plane_src - plane_dst - self.num_planes)
        )
        if plane_diff == 1:
            # 检查是否在极地区域
            lat_src = np.arcsin(self.positions[src][2] / np.linalg.norm(self.positions[src]))
            lat_dst = np.arcsin(self.positions[dst][2] / np.linalg.norm(self.positions[dst]))
            
            # 如果两颗卫星都不在极地区域
            if abs(lat_src) < np.radians(65) and abs(lat_dst) < np.radians(65):
                # 检查运动方向是否相似
                vel_angle = np.dot(self.velocities[src], self.velocities[dst]) / (
                    np.linalg.norm(self.velocities[src]) * np.linalg.norm(self.velocities[dst])
                )
                return vel_angle > 0.7
        
        return False
    
    def _calculate_link_capacity(self, distance: float) -> float:
        """
        计算链路容量
        
        Args:
            distance: 卫星间距离(km)
            
        Returns:
            float: 链路容量(bps)
        """
        # 链路参数
        freq = self.config['link']['frequency']  # 载波频率(Hz)
        tx_power = self.config['link']['transmit_power']  # 发射功率(W)
        noise_temp = self.config['link']['noise_temperature']  # 噪声温度(K)
        bandwidth = self.config['link']['bandwidth']  # 带宽(Hz)
        
        # 计算自由空间路径损耗
        c = 3e8  # 光速(m/s)
        wavelength = c / freq
        path_loss = (4 * np.pi * distance * 1000 / wavelength) ** 2
        
        # 计算接收信噪比
        k = 1.38e-23  # 玻尔兹曼常数
        noise_power = k * noise_temp * bandwidth
        rx_power = tx_power / path_loss
        snr = rx_power / noise_power
        
        # 使用香农公式计算链路容量
        capacity = bandwidth * np.log2(1 + snr)
        
        return float(capacity) 
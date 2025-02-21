from typing import Dict, Any, List
import numpy as np
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
from datetime import datetime, timedelta
from .satellite import Satellite
from math import asin, degrees

class TLEConstellation:
    """基于TLE数据的卫星星座类"""
    
    # 定义轨道面分组
    ORBITAL_PLANES = {
        1: [145, 143, 140, 148, 150, 153, 144, 149, 146, 142, 157],
        2: [134, 141, 137, 116, 135, 151, 120, 113, 138, 130, 131],
        3: [117, 168, 180, 123, 126, 167, 171, 121, 118, 172, 173],
        4: [119, 122, 128, 107, 132, 129, 100, 133, 125, 136, 139],
        5: [158, 160, 159, 163, 165, 166, 154, 164, 108, 155, 156],  # 注意105/164合并
        6: [102, 112, 104, 114, 103, 109, 106, 152, 147, 110, 111]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化TLE星座
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.next_link_update = 0.0  # 初始化链路更新时间
        tle_file = config.get('tle_file', 'tle/Iridium.tle')
        
        # 初始化列表
        self.satellites = []  # 卫星对象列表
        self.sat_names = []  # 卫星名称列表
        
        # 初始化链路缓存
        self.intra_plane_links_cache = {}  # 轨道面内链路缓存
        self.last_cache_update = 0.0  # 上次缓存更新时间
        self.cache_update_interval = 10.0  # 缓存更新间隔（秒）
        
        # 读取TLE文件
        self._load_tle(tle_file)
        
        # 过滤出已分配轨道面的卫星
        valid_satellites = []
        valid_sat_names = []
        valid_sat_indices = {}  # 原始索引到新索引的映射
        
        # 获取所有有效的卫星编号
        valid_sat_numbers = set()
        for satellites in self.ORBITAL_PLANES.values():
            valid_sat_numbers.update(satellites)
        
        # 过滤卫星
        for i, (sat, name) in enumerate(zip(self.satellites, self.sat_names)):
            try:
                sat_number = int(name.split()[1])
                if sat_number in valid_sat_numbers:
                    valid_sat_indices[i] = len(valid_satellites)
                    valid_satellites.append(sat)
                    valid_sat_names.append(name)
            except (ValueError, IndexError):
                continue
        
        # 更新卫星列表
        self.satellites = valid_satellites
        self.sat_names = valid_sat_names
        
        # 初始化位置和速度数组
        self.total_satellites = len(self.satellites)
        self.positions = np.zeros((self.total_satellites, 3))
        self.velocities = np.zeros((self.total_satellites, 3))
        
        # 兼容Walker星座的属性
        self.num_planes = config.get('orbital_planes', 6)
        self.sats_per_plane = config.get('satellites_per_plane', 11)
        self.altitude = config.get('altitude', 781)  # km
        self.inclination = np.radians(config.get('inclination', 86.4))
        
        # 创建Satellite对象
        self.satellite_objects = []
        max_queue_length = config.get('max_queue_length', 100)
        for i in range(self.total_satellites):
            self.satellite_objects.append(Satellite(i, max_queue_length))
            
        # 初始化卫星轨道面映射
        self.satellite_plane_mapping = {}  # 卫星索引到轨道面的映射
        self.satellites_in_planes = {i: [] for i in range(1, 7)}  # 每个轨道面包含的卫星索引列表
        
        # 建立映射关系
        for i in range(self.total_satellites):
            sat_name = self.get_satellite_name(i)
            try:
                sat_number = int(sat_name.split()[1])
                for plane_num, satellites in self.ORBITAL_PLANES.items():
                    if sat_number in satellites:
                        self.satellite_plane_mapping[i] = plane_num
                        # 按照ORBITAL_PLANES中的顺序添加卫星
                        idx = satellites.index(sat_number)
                        self.satellites_in_planes[plane_num].append(i)
                        break
            except:
                continue
        
        # 更新初始状态
        self.update(0.0)
    
    def _load_tle(self, tle_file: str) -> None:
        """
        加载TLE文件
        
        Args:
            tle_file: TLE文件路径
        """
        with open(tle_file, 'r') as f:
            lines = f.readlines()
        
        # 每三行为一组（标题行和两行TLE数据）
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            
            # 使用SGP4创建卫星对象
            satellite = twoline2rv(line1, line2, wgs84)
            self.satellites.append(satellite)
            self.sat_names.append(name)
    
    def update(self, time: float) -> None:
        """
        更新星座状态
        
        Args:
            time: 相对初始时间的时间偏移(s)
        """
        # 计算当前时间
        current_time = datetime.utcnow() + timedelta(seconds=time)
        
        # 更新每颗卫星的位置和速度
        for i, sat in enumerate(self.satellites):
            # 获取位置(km)和速度(km/s)
            position, velocity = sat.propagate(
                current_time.year,
                current_time.month,
                current_time.day,
                current_time.hour,
                current_time.minute,
                current_time.second + current_time.microsecond/1e6
            )
            
            # 更新位置和速度数组
            self.positions[i] = position
            self.velocities[i] = velocity
        
        # 检查是否需要更新缓存
        if time - self.last_cache_update >= self.cache_update_interval:
            self.intra_plane_links_cache.clear()  # 清除旧的缓存
            self.last_cache_update = time
        
        # 更新下一次链路更新时间
        self.next_link_update = time + self.config['simulation']['link']['update_interval']
    
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
            # 计算链路质量
            quality = max(0, 1 - distance / self.config.get('max_range', 5000))
            return {
                'capacity': capacity,
                'delay': delay,
                'distance': distance,
                'quality': quality
            }
        else:
            return {
                'capacity': 0.0,
                'delay': float('inf'),
                'distance': distance,
                'quality': 0.0
            }
    
    def _can_establish_link(self, sat1_idx: int, sat2_idx: int) -> bool:
        """判断两颗卫星是否可以建立链路
        
        参数:
            sat1_idx (int): 第一颗卫星的索引
            sat2_idx (int): 第二颗卫星的索引
            
        返回:
            bool: 如果可以建立链路返回True，否则返回False
        """
        # 获取两颗卫星所在的轨道面
        plane1 = self.satellite_plane_mapping.get(sat1_idx)
        plane2 = self.satellite_plane_mapping.get(sat2_idx)
        
        if plane1 is None or plane2 is None:
            return False
            
        # 获取卫星编号
        try:
            sat1_num = int(self.get_satellite_name(sat1_idx).split()[1])
            sat2_num = int(self.get_satellite_name(sat2_idx).split()[1])
        except (ValueError, IndexError):
            return False
            
        if plane1 == plane2:  # 同一轨道面内的链路
            # 检查缓存
            cache_key = tuple(sorted([sat1_idx, sat2_idx]))
            if cache_key in self.intra_plane_links_cache:
                return self.intra_plane_links_cache[cache_key]
                
            # 获取卫星在ORBITAL_PLANES中的位置
            satellites = self.ORBITAL_PLANES[plane1]
            try:
                idx1 = satellites.index(sat1_num)
                idx2 = satellites.index(sat2_num)
            except ValueError:
                return False
                
            # 计算位置差（考虑环形拓扑）
            n = len(satellites)
            pos_diff = min((idx1 - idx2) % n, (idx2 - idx1) % n)
            
            # 更新缓存
            result = pos_diff == 1  # 只与相邻卫星建立链路
            self.intra_plane_links_cache[cache_key] = result
            return result
            
        else:  # 跨轨道面的链路
            # 只允许相邻轨道面建立链路
            if abs(plane1 - plane2) != 1:
                return False
                
            # 获取卫星位置
            pos1 = self.positions[sat1_idx]
            pos2 = self.positions[sat2_idx]
            
            # 计算纬度（弧度）
            lat1 = np.arcsin(pos1[2] / np.linalg.norm(pos1))
            lat2 = np.arcsin(pos2[2] / np.linalg.norm(pos2))
            
            # 在极地区域（纬度超过±60度）不建立跨轨道面连接
            if abs(np.degrees(lat1)) > 60 or abs(np.degrees(lat2)) > 60:
                return False
                
            # 检查距离约束
            distance = np.linalg.norm(pos1 - pos2)
            if distance < 1500 or distance > 5000:
                return False
                
            # 获取两颗卫星在各自轨道面中的位置
            satellites1 = self.ORBITAL_PLANES[plane1]
            satellites2 = self.ORBITAL_PLANES[plane2]
            try:
                idx1 = satellites1.index(sat1_num)
                idx2 = satellites2.index(sat2_num)
            except ValueError:
                return False
                
            # 计算相对位置差
            n = len(satellites1)  # 假设所有轨道面卫星数量相同
            relative_pos1 = idx1 / n  # 归一化位置 [0,1]
            relative_pos2 = idx2 / n
            
            # 考虑环形拓扑的相对位置差
            pos_diff = min(
                abs(relative_pos1 - relative_pos2),
                abs(relative_pos1 - relative_pos2 + 1),
                abs(relative_pos1 - relative_pos2 - 1)
            )
            
            # 判断是否是最近的卫星
            # 允许与相邻轨道面上最近的两颗卫星建立连接
            return pos_diff <= 2.0 / n  # 允许连接最近的两颗卫星
    
    def _calculate_link_capacity(self, distance: float) -> float:
        """
        计算链路容量
        
        Args:
            distance: 卫星间距离(km)
            
        Returns:
            float: 链路容量(bps)
        """
        # 链路参数
        freq = float(self.config.get('link', {}).get('frequency', 26e9))  # 载波频率(Hz)
        tx_power = float(self.config.get('link', {}).get('transmit_power', 20))  # 发射功率(W)
        noise_temp = float(self.config.get('link', {}).get('noise_temperature', 290))  # 噪声温度(K)
        bandwidth = float(self.config.get('link', {}).get('bandwidth', 1e9))  # 带宽(Hz)
        
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
    
    def get_satellite_name(self, sat_id: int) -> str:
        """
        获取卫星名称
        
        Args:
            sat_id: 卫星ID
            
        Returns:
            str: 卫星名称
        """
        return self.sat_names[sat_id]
    
    def reset(self) -> None:
        """重置星座状态"""
        # 重置位置和速度
        self.positions = np.zeros((self.total_satellites, 3))
        self.velocities = np.zeros((self.total_satellites, 3))
        
        # 重置每个卫星的状态
        for sat in self.satellite_objects:
            sat.reset()
        
        # 更新初始状态
        self.update(0.0)
    
    def get_valid_neighbors(self, sat_id: int) -> List[int]:
        """
        获取卫星的有效邻居节点
        
        Args:
            sat_id: 卫星ID
            
        Returns:
            List[int]: 有效邻居节点列表
        """
        valid_neighbors = []
        
        # 获取当前卫星所在的轨道面
        current_plane = self.satellite_plane_mapping.get(sat_id)
        if current_plane is None:
            return []
        
        # 检查同一轨道面内的相邻卫星
        satellites_in_plane = self.satellites_in_planes[current_plane]
        current_idx = satellites_in_plane.index(sat_id)
        sats_per_plane = len(satellites_in_plane)
        
        # 检查前一个卫星
        prev_idx = (current_idx - 1) % sats_per_plane
        prev_sat = satellites_in_plane[prev_idx]
        if self._can_establish_link(sat_id, prev_sat):
            valid_neighbors.append(prev_sat)
        
        # 检查后一个卫星
        next_idx = (current_idx + 1) % sats_per_plane
        next_sat = satellites_in_plane[next_idx]
        if self._can_establish_link(sat_id, next_sat):
            valid_neighbors.append(next_sat)
        
        # 检查相邻轨道面的卫星
        current_plane_num = current_plane
        for plane_diff in [-1, 1]:
            adjacent_plane = current_plane_num + plane_diff
            if adjacent_plane in self.satellites_in_planes:
                # 获取相邻轨道面的卫星
                adjacent_sats = self.satellites_in_planes[adjacent_plane]
                for adj_sat in adjacent_sats:
                    if self._can_establish_link(sat_id, adj_sat):
                        valid_neighbors.append(adj_sat)
        
        return valid_neighbors
    
    def get_link_quality(self, sat1_idx: int, sat2_idx: int) -> float:
        """
        获取两颗卫星之间的链路质量
        
        Args:
            sat1_idx: 第一颗卫星的索引
            sat2_idx: 第二颗卫星的索引
            
        Returns:
            float: 链路质量，范围[0,1]，0表示无法建立链路
        """
        if not self._can_establish_link(sat1_idx, sat2_idx):
            return 0.0
        
        # 计算卫星间距离
        pos1 = self.positions[sat1_idx]
        pos2 = self.positions[sat2_idx]
        distance = np.linalg.norm(pos1 - pos2)
        
        # 根据距离计算链路质量
        max_range = float(self.config.get('max_range', 5000))  # 最大通信距离
        min_range = float(self.config.get('min_range', 2000))  # 最小通信距离
        
        if distance < min_range or distance > max_range:
            return 0.0
        
        # 线性映射到[0,1]区间
        quality = 1.0 - (distance - min_range) / (max_range - min_range)
        return max(0.0, min(1.0, quality)) 
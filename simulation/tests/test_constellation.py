import unittest
import numpy as np
from simulation.topology import WalkerConstellation

class TestWalkerConstellation(unittest.TestCase):
    """测试Walker星座实现"""
    
    def setUp(self):
        """测试前的设置"""
        self.config = {
            'orbital_planes': 6,
            'satellites_per_plane': 11,
            'altitude': 550,
            'inclination': 53.0,
            'max_range': 4000,  # 增加最大通信范围
            'link': {
                'frequency': 26e9,
                'transmit_power': 20,  # 增加发射功率
                'noise_temperature': 290,
                'bandwidth': 1e9
            }
        }
        self.constellation = WalkerConstellation(self.config)
    
    def test_initialization(self):
        """测试星座初始化"""
        # 检查卫星数量
        self.assertEqual(self.constellation.total_satellites, 66)
        
        # 检查位置和速度数组维度
        self.assertEqual(self.constellation.positions.shape, (66, 3))
        self.assertEqual(self.constellation.velocities.shape, (66, 3))
        
        # 验证所有卫星的轨道高度
        for i in range(66):
            pos = self.constellation.positions[i]
            height = np.linalg.norm(pos) - 6371.0  # 减去地球半径
            self.assertAlmostEqual(height, 550.0, places=1)
    
    def test_satellite_distribution(self):
        """测试卫星分布"""
        # 验证每个轨道面的卫星数量
        for plane in range(self.config['orbital_planes']):
            plane_sats = [i for i in range(66) if i // 11 == plane]
            self.assertEqual(len(plane_sats), 11)
            
            # 验证相邻卫星之间的距离
            for i in range(len(plane_sats)-1):
                sat1 = plane_sats[i]
                sat2 = plane_sats[i+1]
                distance = np.linalg.norm(
                    self.constellation.positions[sat1] - 
                    self.constellation.positions[sat2]
                )
                self.assertLess(distance, self.config['max_range'])
    
    def test_link_establishment(self):
        """测试链路建立"""
        # 测试同一轨道面内的相邻卫星
        self.assertTrue(self.constellation._can_establish_link(0, 1))
        self.assertTrue(self.constellation._can_establish_link(1, 2))
        
        # 测试不同轨道面的卫星
        plane_size = self.config['satellites_per_plane']
        self.assertFalse(self.constellation._can_establish_link(0, plane_size+1))
        
        # 测试极地区域的链路
        # 找到接近极地的卫星
        polar_sat = None
        for i in range(66):
            pos = self.constellation.positions[i]
            lat = np.arcsin(pos[2] / np.linalg.norm(pos))
            if abs(lat) > np.radians(65):
                polar_sat = i
                break
                
        if polar_sat is not None:
            # 验证极地区域不建立跨平面链路
            next_plane_sat = polar_sat + plane_size
            if next_plane_sat < 66:
                self.assertFalse(
                    self.constellation._can_establish_link(polar_sat, next_plane_sat)
                )
    
    def test_link_capacity(self):
        """测试链路容量计算"""
        # 测试不同距离的链路容量
        distances = [100, 500, 1000]
        capacities = [
            self.constellation._calculate_link_capacity(d) 
            for d in distances
        ]
        
        # 验证容量随距离增加而减小
        self.assertTrue(all(c1 > c2 for c1, c2 in zip(capacities, capacities[1:])))
        
        # 验证超出最大范围的链路
        link_data = self.constellation.get_link_data(0, 33)  # 选择较远的卫星
        if link_data['distance'] > self.config['max_range']:
            self.assertEqual(link_data['capacity'], 0.0)
            self.assertEqual(link_data['quality'], 0.0)
    
    def test_constellation_update(self):
        """测试星座更新"""
        # 记录初始位置
        initial_positions = self.constellation.positions.copy()
        
        # 更新星座状态
        self.constellation.update(100.0)  # 更新100秒
        
        # 验证位置已更新
        self.assertFalse(np.array_equal(initial_positions, self.constellation.positions))
        
        # 验证更新后的轨道高度保持不变
        for i in range(66):
            pos = self.constellation.positions[i]
            height = np.linalg.norm(pos) - 6371.0
            self.assertAlmostEqual(height, 550.0, places=1)
    
    def test_link_data(self):
        """测试链路数据获取"""
        # 测试相邻卫星的链路数据
        link_data = self.constellation.get_link_data(0, 1)
        self.assertGreater(link_data['capacity'], 0.0)
        self.assertGreater(link_data['quality'], 0.0)
        self.assertLess(link_data['delay'], float('inf'))
        
        # 测试不可连接卫星的链路数据
        distant_sat = 33  # 选择较远的卫星
        link_data = self.constellation.get_link_data(0, distant_sat)
        if not self.constellation._can_establish_link(0, distant_sat):
            self.assertEqual(link_data['capacity'], 0.0)
            self.assertEqual(link_data['quality'], 0.0)
            self.assertEqual(link_data['delay'], float('inf'))

if __name__ == '__main__':
    unittest.main() 
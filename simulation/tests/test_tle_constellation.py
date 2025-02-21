import unittest
import numpy as np
from simulation.tle_constellation import TLEConstellation

class TestTLEConstellation(unittest.TestCase):
    """测试TLE星座类"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建配置字典
        self.config = {
            'tle_file': 'tle/Iridium.tle',
            'orbital_planes': 6,
            'satellites_per_plane': 11,
            'max_range': 5000,
            'link': {
                'frequency': 23e9,
                'transmit_power': 10.0,
                'noise_temperature': 290,
                'bandwidth': 1e9,
                'max_latitude': 80.0
            }
        }
        
        # 创建星座对象
        self.constellation = TLEConstellation(self.config)
    
    def test_get_valid_neighbors(self):
        """测试获取有效邻居节点的方法"""
        # 测试几个不同轨道面的卫星
        test_cases = [
            # 轨道面1的卫星
            (40, 'IRIDIUM 140'),  # 轨道面1
            # 轨道面2的卫星
            (34, 'IRIDIUM 141'),  # 轨道面2
            # 轨道面5的卫星
            (50, 'IRIDIUM 160'),  # 轨道面5
            # 轨道面6的卫星
            (0, 'IRIDIUM 106'),   # 轨道面6
        ]

        for sat_idx, expected_name in test_cases:
            # 验证卫星名称
            self.assertEqual(self.constellation.get_satellite_name(sat_idx), expected_name)
            
            # 获取有效邻居节点
            valid_neighbors = self.constellation.get_valid_neighbors(sat_idx)
            
            # 打印调试信息
            print(f"\n测试卫星 {expected_name} (索引 {sat_idx}) 的有效邻居:")
            for neighbor in valid_neighbors:
                print(f"- {self.constellation.get_satellite_name(neighbor)} (索引 {neighbor})")
            
            # 验证每个邻居的有效性
            for neighbor in valid_neighbors:
                # 验证链路状态
                link_quality = self.constellation.get_link_quality(sat_idx, neighbor)
                self.assertGreater(link_quality, 0, 
                    f"链路质量应大于0: {expected_name} -> {self.constellation.get_satellite_name(neighbor)}")
                
                # 获取当前卫星和邻居卫星的轨道面
                current_plane = self.constellation.satellite_plane_mapping.get(sat_idx)
                neighbor_plane = self.constellation.satellite_plane_mapping.get(neighbor)
                
                # 验证轨道面关系
                plane_diff = abs(current_plane - neighbor_plane)
                self.assertLessEqual(plane_diff, 1, 
                    f"邻居卫星应在相同或相邻轨道面: {expected_name} -> {self.constellation.get_satellite_name(neighbor)}")
                
                if current_plane == neighbor_plane:
                    # 在同一轨道面内，验证是否相邻
                    plane_sats = self.constellation.satellites_in_planes[current_plane]
                    current_pos = plane_sats.index(sat_idx)
                    neighbor_pos = plane_sats.index(neighbor)
                    pos_diff = min((current_pos - neighbor_pos) % len(plane_sats),
                                 (neighbor_pos - current_pos) % len(plane_sats))
                    self.assertEqual(pos_diff, 1,
                        f"同一轨道面内的卫星应相邻: {expected_name} -> {self.constellation.get_satellite_name(neighbor)}")

    def test_print_satellite_mapping(self):
        """打印卫星索引映射关系"""
        print("\n卫星索引映射关系:")
        for i in range(self.constellation.total_satellites):
            sat_name = self.constellation.get_satellite_name(i)
            plane = self.constellation.satellite_plane_mapping.get(i)
            print(f"索引 {i}: {sat_name} -> 轨道面 {plane}")
            
        print("\n轨道面内卫星分布:")
        for plane_num, satellites in self.constellation.satellites_in_planes.items():
            print(f"\n轨道面 {plane_num}:")
            for sat_idx in satellites:
                sat_name = self.constellation.get_satellite_name(sat_idx)
                print(f"  - 索引 {sat_idx}: {sat_name}")

if __name__ == '__main__':
    # 创建测试套件
    suite = unittest.TestSuite()
    # 添加测试用例
    suite.addTest(TestTLEConstellation('test_print_satellite_mapping'))
    # 运行测试
    unittest.TextTestRunner(verbosity=2).run(suite) 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from simulation.topology import WalkerConstellation

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_constellation(constellation: WalkerConstellation):
    """可视化卫星星座"""
    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地球
    r_earth = 6371  # 地球半径(km)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    
    # 获取所有卫星位置
    positions = constellation.positions
    
    # 为每个轨道面使用不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, constellation.num_planes))
    
    # 绘制轨道和卫星
    for plane in range(constellation.num_planes):
        # 获取该轨道面的卫星
        plane_sats = positions[plane * constellation.sats_per_plane:(plane + 1) * constellation.sats_per_plane]
        
        # 绘制卫星
        ax.scatter(plane_sats[:, 0], plane_sats[:, 1], plane_sats[:, 2], 
                  c=[colors[plane]], marker='o', s=50, label=f'Plane {plane+1}')
        
        # 生成更多点来绘制平滑的轨道路径
        num_points = 200
        t = np.linspace(0, 2*np.pi, num_points)
        r = constellation.altitude + 6371.0  # 轨道半径
        
        # 计算RAAN
        raan = 2 * np.pi * plane / constellation.num_planes
        
        # 在赤道面内的圆形轨道
        x_eq = r * np.cos(t)
        y_eq = r * np.sin(t)
        z_eq = np.zeros_like(t)
        
        # 绕x轴旋转（倾角）
        x_incl = x_eq
        y_incl = y_eq * np.cos(constellation.inclination)
        z_incl = y_eq * np.sin(constellation.inclination)
        
        # 绕z轴旋转（RAAN）
        x_orbit = x_incl * np.cos(raan) - y_incl * np.sin(raan)
        y_orbit = x_incl * np.sin(raan) + y_incl * np.cos(raan)
        z_orbit = z_incl
        
        # 绘制轨道路径
        ax.plot(x_orbit, y_orbit, z_orbit, color=colors[plane], alpha=0.5, linewidth=1.5)
    
    # 绘制卫星间链路
    for i in range(constellation.total_satellites):
        for j in range(i+1, constellation.total_satellites):
            if constellation._can_establish_link(i, j):
                pos_i = positions[i]
                pos_j = positions[j]
                ax.plot([pos_i[0], pos_j[0]],
                       [pos_i[1], pos_j[1]],
                       [pos_i[2], pos_j[2]], 'gray', alpha=0.3, linewidth=0.5)
    
    # 设置图形属性
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Walker Delta Constellation (86.4°:66/6/2)')
    
    # 设置坐标轴范围
    max_range = r_earth + constellation.altitude + 500
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # 设置等比例显示
    ax.set_box_aspect([1,1,1])
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 设置更好的视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 加载配置
    config = load_config('config/config.yaml')
    
    # 创建星座
    constellation = WalkerConstellation(config['topology'])
    
    # 更新星座状态(t=0)
    constellation.update(0.0)
    
    # 可视化星座
    visualize_constellation(constellation)
    
    # 打印星座信息
    print("\n星座信息:")
    print(f"轨道面数量: {constellation.num_planes}")
    print(f"每个轨道面的卫星数: {constellation.sats_per_plane}")
    print(f"总卫星数: {constellation.total_satellites}")
    print(f"轨道高度: {constellation.altitude} km")
    print(f"轨道倾角: {np.degrees(constellation.inclination)}°")
    
    # 检查链路连接性
    total_links = 0
    for i in range(constellation.total_satellites):
        for j in range(i+1, constellation.total_satellites):
            if constellation._can_establish_link(i, j):
                total_links += 1
    
    print(f"\n总链路数: {total_links}")
    print(f"平均每颗卫星的链路数: {2 * total_links / constellation.total_satellites:.2f}")

if __name__ == "__main__":
    main() 
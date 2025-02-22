import numpy as np
import plotly.graph_objects as go
import yaml
from simulation.topology import WalkerConstellation

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_earth_sphere():
    """创建地球球体"""
    radius = 6371.0  # 地球半径(km)
    phi = np.linspace(0, 2*np.pi, 100)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    phi, theta = np.meshgrid(phi, theta)
    
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.sin(theta)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        showscale=False,
        name='Earth'
    )

def visualize_constellation():
    """可视化卫星星座"""
    # 创建星座对象
    config = load_config('config/config.yaml')
    constellation = WalkerConstellation(config['topology'])
    
    # 更新卫星位置
    constellation.update(0.0)
    
    # 创建图形
    fig = go.Figure()
    
    # 添加地球
    fig.add_trace(create_earth_sphere())
    
    # 为每个轨道面使用不同的颜色
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    # 轨道半径
    r = constellation.altitude + 6371.0  # 轨道半径(km)
    
    # 为每个轨道平面绘制卫星和轨道
    for plane in range(constellation.num_planes):
        # 计算该平面的RAAN
        raan = 2 * np.pi * plane / constellation.num_planes
        
        # 生成轨道路径点
        t = np.linspace(0, 2*np.pi, 200)
        
        # 1. 在赤道面内生成圆形轨道
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros_like(t)
        
        # 2. 绕x轴旋转（倾角）
        inclination = constellation.inclination
        x_incl = x
        y_incl = y * np.cos(inclination) - z * np.sin(inclination)
        z_incl = y * np.sin(inclination) + z * np.cos(inclination)
        
        # 3. 绕z轴旋转（RAAN）
        x_final = x_incl * np.cos(raan) - y_incl * np.sin(raan)
        y_final = x_incl * np.sin(raan) + y_incl * np.cos(raan)
        z_final = z_incl
        
        # 绘制轨道路径
        fig.add_trace(go.Scatter3d(
            x=x_final,
            y=y_final,
            z=z_final,
            mode='lines',
            line=dict(
                color=colors[plane % len(colors)],
                width=2
            ),
            name=f'Orbit {plane+1}'
        ))
        
        # 获取该轨道面的卫星
        satellites_in_plane = []
        for sat in range(constellation.sats_per_plane):
            sat_id = plane * constellation.sats_per_plane + sat
            pos = constellation.positions[sat_id]
            satellites_in_plane.append(pos)
        
        satellites_in_plane = np.array(satellites_in_plane)
        
        # 绘制卫星
        fig.add_trace(go.Scatter3d(
            x=satellites_in_plane[:, 0],
            y=satellites_in_plane[:, 1],
            z=satellites_in_plane[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors[plane % len(colors)],
            ),
            name=f'Plane {plane+1} Satellites'
        ))
        
        # 绘制卫星间链路
        for i in range(constellation.sats_per_plane):
            sat_id = plane * constellation.sats_per_plane + i
            for j in range(i+1, constellation.sats_per_plane):
                next_sat_id = plane * constellation.sats_per_plane + j
                if constellation._can_establish_link(sat_id, next_sat_id):
                    pos1 = constellation.positions[sat_id]
                    pos2 = constellation.positions[next_sat_id]
                    fig.add_trace(go.Scatter3d(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        z=[pos1[2], pos2[2]],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False
                    ))
    
    # 设置布局
    fig.update_layout(
        title='Walker星座3D可视化',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.8,
            y=0.9
        )
    )
    
    # 保存为HTML文件
    fig.write_html('constellation_3d.html')
    
    # 统计链路数量
    total_links = 0
    for i in range(constellation.total_satellites):
        for j in range(i+1, constellation.total_satellites):
            if constellation._can_establish_link(i, j):
                total_links += 1
    
    print("\n星座信息:")
    print(f"轨道面数量: {constellation.num_planes}")
    print(f"每个轨道面的卫星数: {constellation.sats_per_plane}")
    print(f"总卫星数: {constellation.total_satellites}")
    print(f"轨道高度: {constellation.altitude} km")
    print(f"轨道倾角: {np.degrees(constellation.inclination)}°")
    print(f"\n总链路数: {total_links}")
    print(f"平均每颗卫星的链路数: {2*total_links/constellation.total_satellites:.2f}")

def test_orbital_planes():
    """测试轨道面分布"""
    # 创建星座对象
    config = load_config('config/config.yaml')
    constellation = WalkerConstellation(config['topology'])
    
    # 更新卫星位置
    constellation.update(0.0)
    
    # 创建图形
    fig = go.Figure()
    
    # 添加地球
    fig.add_trace(create_earth_sphere())
    
    # 轨道半径
    r = constellation.altitude + 6371.0  # 轨道半径(km)
    
    # 手动设置6个轨道面的RAAN值
    raan_values = [i * 2 * np.pi / 6 for i in range(6)]  # 6个轨道面，每个间隔60度
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    # 为每个轨道平面单独绘制
    for i, raan in enumerate(raan_values):
        # 生成轨道路径点
        t = np.linspace(0, 2*np.pi, 200)
        
        # 1. 在轨道平面内生成圆形轨道
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros_like(t)
        
        # 2. 绕x轴旋转（倾角）
        x_incl = x
        y_incl = y * np.cos(constellation.inclination) - z * np.sin(constellation.inclination)
        z_incl = y * np.sin(constellation.inclination) + z * np.cos(constellation.inclination)
        
        # 3. 绕z轴旋转（RAAN）
        x_final = x_incl * np.cos(raan) - y_incl * np.sin(raan)
        y_final = x_incl * np.sin(raan) + y_incl * np.cos(raan)
        z_final = z_incl
        
        # 绘制轨道路径
        fig.add_trace(go.Scatter3d(
            x=x_final,
            y=y_final,
            z=z_final,
            mode='lines',
            line=dict(
                color=colors[i],
                width=2
            ),
            name=f'轨道面 {i+1} (RAAN={np.degrees(raan):.1f}°)'
        ))
        
        # 在轨道上均匀分布卫星
        sat_positions = np.linspace(0, 2*np.pi, constellation.sats_per_plane)
        satellites_x = r * np.cos(sat_positions)
        satellites_y = r * np.sin(sat_positions)
        satellites_z = np.zeros_like(sat_positions)
        
        # 应用相同的旋转到卫星位置
        # 1. 倾角旋转（绕x轴）
        satellites_x_incl = satellites_x
        satellites_y_incl = satellites_y * np.cos(constellation.inclination) - satellites_z * np.sin(constellation.inclination)
        satellites_z_incl = satellites_y * np.sin(constellation.inclination) + satellites_z * np.cos(constellation.inclination)
        
        # 2. RAAN旋转（绕z轴）
        satellites_x_final = satellites_x_incl * np.cos(raan) - satellites_y_incl * np.sin(raan)
        satellites_y_final = satellites_x_incl * np.sin(raan) + satellites_y_incl * np.cos(raan)
        satellites_z_final = satellites_z_incl
        
        # 绘制卫星
        fig.add_trace(go.Scatter3d(
            x=satellites_x_final,
            y=satellites_y_final,
            z=satellites_z_final,
            mode='markers',
            marker=dict(
                size=5,
                color=colors[i],
            ),
            name=f'卫星 (轨道面 {i+1})'
        ))
    
    # 设置布局
    fig.update_layout(
        title='Walker星座轨道面分布 (6个轨道面)',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)  # 调整视角以更好地观察轨道面
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.8,
            y=0.9
        )
    )
    
    # 保存为HTML文件
    fig.write_html('orbital_planes_test.html')
    
    # 打印轨道面信息
    print("\n轨道面分布:")
    for i, raan in enumerate(raan_values):
        print(f"轨道面 {i+1}: RAAN = {np.degrees(raan):.1f}°")

def main():
    """主函数"""
    # 运行轨道面测试
    test_orbital_planes()
    
    print("\n请在浏览器中打开 orbital_planes_test.html 查看结果")
    print("注意观察:")
    print("1. 轨道面是否均匀分布（相邻轨道面之间的角度应该相等）")
    print("2. 从顶部视图观察时，轨道面是否呈现对称的星形图案")
    print("3. 每个轨道面内的卫星是否均匀分布")

if __name__ == "__main__":
    main() 
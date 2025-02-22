import numpy as np
import plotly.graph_objects as go
from simulation.tle_constellation import TLEConstellation

# 定义轨道面分组
ORBITAL_PLANES = {
    1: [145, 143, 140, 148, 150, 153, 144, 149, 146, 142, 157],
    2: [134, 141, 137, 116, 135, 151, 120, 113, 138, 130, 131],
    3: [117, 168, 180, 123, 126, 167, 171, 121, 118, 172, 173],
    4: [119, 122, 128, 107, 132, 129, 100, 133, 125, 136, 139],
    5: [158, 160, 159, 163, 165, 166, 154, 164, 108, 155, 156],  # 注意105/164合并
    6: [102, 112, 104, 114, 103, 109, 106, 152, 147, 110, 111]
}

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

def get_orbital_plane(sat_name: str) -> int:
    """获取卫星所属的轨道面编号"""
    # 从卫星名称中提取编号
    try:
        sat_number = int(sat_name.split()[1])
        for plane_num, satellites in ORBITAL_PLANES.items():
            if sat_number in satellites:
                return plane_num
    except:
        pass
    return 0  # 如果找不到对应的轨道面，返回0

def visualize_tle_constellation(tle_file: str):
    """可视化TLE星座"""
    # 创建配置字典
    config = {
        'tle_file': tle_file,
        'orbital_planes': 6,
        'satellites_per_plane': 11,
        'max_range': 5000,
        'simulation': {
            'network': {
                'max_buffer_size': 20,
                'max_queue_length': 20,
                'total_satellites': 66
            },
            'link': {
                'update_interval': 1.0,
                'buffer_size': 20,
                'bandwidth': 10e6,
                'frequency': 23e9,
                'transmit_power': 10.0,
                'noise_temperature': 290,
                'bandwidth': 1e9,
                'path_loss_model': {
                    'type': 'free_space'
                }
            }
        }
    }
    
    # 创建星座对象
    constellation = TLEConstellation(config)
    
    # 统计每个轨道面的卫星
    plane_stats = {i: [] for i in range(1, 7)}
    unassigned = []
    
    for i in range(constellation.total_satellites):
        sat_name = constellation.get_satellite_name(i)
        plane_num = get_orbital_plane(sat_name)
        if plane_num > 0:
            plane_stats[plane_num].append(sat_name)
        else:
            unassigned.append(sat_name)
    
    # 打印统计信息
    print("\n轨道面分配统计:")
    total_assigned = 0
    for plane_num, satellites in plane_stats.items():
        print(f"轨道面 {plane_num}: {len(satellites)}颗卫星")
        total_assigned += len(satellites)
    if unassigned:
        print(f"\n未分配的卫星: {len(unassigned)}颗")
        for sat in unassigned:
            print(f"  - {sat}")
    print(f"\n总计: {total_assigned}颗已分配卫星")
    
    # 创建图形
    fig = go.Figure()
    
    # 添加地球
    fig.add_trace(create_earth_sphere())
    
    # 颜色设置 - 每个轨道面一个颜色
    colors = {
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'yellow',
        5: 'purple',
        6: 'orange'
    }
    
    # 为每个轨道面创建轨迹点集合
    plane_positions = {i: [] for i in range(1, 7)}
    
    # 绘制卫星并收集轨道面位置
    for i in range(constellation.total_satellites):
        pos = constellation.positions[i]
        vel = constellation.velocities[i]
        
        # 获取卫星名称和所属轨道面
        sat_name = constellation.get_satellite_name(i)
        plane_num = get_orbital_plane(sat_name)
        
        if plane_num > 0:  # 只处理已知轨道面的卫星
            # 收集轨道面位置
            plane_positions[plane_num].append(pos)
            
            # 绘制卫星
            fig.add_trace(go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[plane_num],
                ),
                name=f'{sat_name} (Plane {plane_num})'
            ))
    
    # 绘制轨道面
    for plane_num, positions in plane_positions.items():
        if positions:
            # 将位置点转换为numpy数组
            positions = np.array(positions)
            
            # 按照角度排序点
            center = np.mean(positions, axis=0)
            angles = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_positions = positions[sorted_indices]
            
            # 闭合轨道
            positions_loop = np.vstack([sorted_positions, sorted_positions[0]])
            
            # 绘制轨道线
            fig.add_trace(go.Scatter3d(
                x=positions_loop[:, 0],
                y=positions_loop[:, 1],
                z=positions_loop[:, 2],
                mode='lines',
                line=dict(
                    color=colors[plane_num],
                    width=1,
                    dash='dot'  # 轨道线使用点线
                ),
                name=f'Orbital Plane {plane_num}'
            ))
    
    # 绘制卫星间链路
    for i in range(constellation.total_satellites):
        sat_name_i = constellation.get_satellite_name(i)
        plane_i = get_orbital_plane(sat_name_i)
        
        if plane_i == 0:  # 跳过未分配轨道面的卫星
            continue
            
        for j in range(i+1, constellation.total_satellites):
            sat_name_j = constellation.get_satellite_name(j)
            plane_j = get_orbital_plane(sat_name_j)
            
            if plane_j == 0:  # 跳过未分配轨道面的卫星
                continue
                
            if constellation._can_establish_link(i, j):
                pos_i = constellation.positions[i]
                pos_j = constellation.positions[j]
                
                # 根据链路类型设置不同的样式
                if plane_i == plane_j:
                    # 同一轨道面内的链路 - 使用实线
                    line_color = colors[plane_i]
                    line_width = 2
                    line_dash = 'solid'
                else:
                    # 跨轨道面的链路 - 使用虚线
                    line_color = 'gray'
                    line_width = 1
                    line_dash = 'dash'
                
                fig.add_trace(go.Scatter3d(
                    x=[pos_i[0], pos_j[0]],
                    y=[pos_i[1], pos_j[1]],
                    z=[pos_i[2], pos_j[2]],
                    mode='lines',
                    line=dict(
                        color=line_color,
                        width=line_width,
                        dash=line_dash
                    ),
                    showlegend=False
                ))
    
    # 设置两个不同的视角
    camera_views = [
        # 3D视角
        dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=2.0)
        ),
        # 俯视图
        dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=2.5)
        )
    ]
    
    # 添加按钮切换视角
    fig.update_layout(
        title='Iridium卫星星座分布 (6轨道面)',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        ),
        showlegend=True,
        legend=dict(
            x=0.8,
            y=0.9,
            groupclick="toggleitem"
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(
                    label='3D视图',
                    method='relayout',
                    args=['scene.camera', camera_views[0]]
                ),
                dict(
                    label='俯视图',
                    method='relayout',
                    args=['scene.camera', camera_views[1]]
                )
            ],
            x=0.9,
            y=1.0
        )]
    )
    
    # 保存为HTML文件
    fig.write_html('iridium_constellation.html')

def main():
    """主函数"""
    # 创建配置字典
    config = {
        'tle_file': 'tle/Iridium.tle',
        'orbital_planes': 6,
        'satellites_per_plane': 11,
        'max_range': 5000,
        'simulation': {
            'network': {
                'max_buffer_size': 20,
                'max_queue_length': 20,
                'total_satellites': 66
            },
            'link': {
                'update_interval': 1.0,
                'buffer_size': 20,
                'bandwidth': 10e6,
                'frequency': 23e9,
                'transmit_power': 10.0,
                'noise_temperature': 290,
                'bandwidth': 1e9,
                'path_loss_model': {
                    'type': 'free_space'
                }
            }
        }
    }
    
    # 运行TLE星座可视化
    visualize_tle_constellation('tle/Iridium.tle')
    
    print("\n请在浏览器中打开 iridium_constellation.html 查看结果")
    print("注意观察:")
    print("1. 六个轨道面的分布情况（每个轨道面用不同颜色标识）")
    print("2. 每个轨道面内的卫星分布")
    print("3. 同一轨道面内的链路（实线）和跨轨道面的链路（虚线）")
    
    # 创建星座对象用于统计链接
    constellation = TLEConstellation(config)
    
    # 分析轨道面5的链路情况
    print("\n轨道面5详细分析:")
    plane5_sats = ORBITAL_PLANES[5]
    print(f"卫星编号: {plane5_sats}")
    
    # 检查轨道面5内部的链路
    print("\n轨道面5内部链路:")
    for i in range(len(plane5_sats)):
        sat1_num = plane5_sats[i]
        sat2_num = plane5_sats[(i + 1) % len(plane5_sats)]  # 环形连接
        
        # 找到卫星在星座中的索引
        sat1_idx = None
        sat2_idx = None
        for idx in range(constellation.total_satellites):
            sat_name = constellation.get_satellite_name(idx)
            try:
                sat_num = int(sat_name.split()[1])
                if sat_num == sat1_num:
                    sat1_idx = idx
                if sat_num == sat2_num:
                    sat2_idx = idx
            except:
                continue
        
        if sat1_idx is not None and sat2_idx is not None:
            can_link = constellation._can_establish_link(sat1_idx, sat2_idx)
            if can_link:
                distance = np.linalg.norm(constellation.positions[sat1_idx] - constellation.positions[sat2_idx])
                lat1 = np.arcsin(constellation.positions[sat1_idx][2] / np.linalg.norm(constellation.positions[sat1_idx]))
                lat2 = np.arcsin(constellation.positions[sat2_idx][2] / np.linalg.norm(constellation.positions[sat2_idx]))
                print(f"卫星 {sat1_num} -> {sat2_num}: 可连接")
                print(f"  距离: {distance:.2f}km")
                print(f"  纬度: {np.degrees(lat1):.2f}°, {np.degrees(lat2):.2f}°")
            else:
                print(f"卫星 {sat1_num} -> {sat2_num}: 不可连接")
                if sat1_idx is not None and sat2_idx is not None:
                    distance = np.linalg.norm(constellation.positions[sat1_idx] - constellation.positions[sat2_idx])
                    lat1 = np.arcsin(constellation.positions[sat1_idx][2] / np.linalg.norm(constellation.positions[sat1_idx]))
                    lat2 = np.arcsin(constellation.positions[sat2_idx][2] / np.linalg.norm(constellation.positions[sat2_idx]))
                    print(f"  距离: {distance:.2f}km")
                    print(f"  纬度: {np.degrees(lat1):.2f}°, {np.degrees(lat2):.2f}°")
    
    # 统计同一轨道面内的链接
    intra_plane_links = {i: 0 for i in range(1, 7)}
    # 统计跨轨道面的链接
    inter_plane_links = 0
    # 统计总链接数
    total_links = 0
    
    # 遍历所有可能的卫星对
    for i in range(constellation.total_satellites):
        sat_name_i = constellation.get_satellite_name(i)
        plane_i = get_orbital_plane(sat_name_i)
        
        for j in range(i+1, constellation.total_satellites):
            sat_name_j = constellation.get_satellite_name(j)
            plane_j = get_orbital_plane(sat_name_j)
            
            if constellation._can_establish_link(i, j):
                total_links += 1
                if plane_i > 0 and plane_j > 0:  # 只统计已知轨道面的卫星
                    if plane_i == plane_j:
                        intra_plane_links[plane_i] += 1
                    else:
                        inter_plane_links += 1
    
    print("\n链接统计信息:")
    print("同一轨道面内的链接:")
    for plane, count in intra_plane_links.items():
        print(f"  轨道面 {plane}: {count}条链接")
    print(f"跨轨道面的链接: {inter_plane_links}条")
    print(f"总链接数: {total_links}条")
    print(f"平均每颗卫星的链接数: {2 * total_links / constellation.total_satellites:.2f}条")

if __name__ == "__main__":
    main() 
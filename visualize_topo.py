import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# 使用与STK相同的参数
ORBIT_HEIGHT_LEO = 1500  # LEO轨道高度(km)
ORBIT_HEIGHT_MEO = 8000  # MEO轨道高度(km)
NUM_ORBITS_LEO = 16     # LEO轨道面数量
SATS_PER_ORBIT_LEO = 16 # 每个LEO轨道面的卫星数量
NUM_ORBITS_MEO = 2      # MEO轨道面数量
SATS_PER_ORBIT_MEO = 8  # 每个MEO轨道面的卫星数量
INCLINATION = 55        # 轨道倾角(度)
EARTH_RADIUS = 6371     # 地球半径(km)

def calculate_satellite_position(orbit, pos, height, inclination, num_orbits, sats_per_orbit):
    """计算卫星的3D坐标"""
    # 计算轨道平面的角度
    orbit_angle = (orbit * 360.0 / num_orbits)
    # 计算卫星在轨道上的位置角度
    pos_angle = (pos * 360.0 / sats_per_orbit)
    
    # 转换为弧度
    orbit_rad = math.radians(orbit_angle)
    pos_rad = math.radians(pos_angle)
    incl_rad = math.radians(inclination)
    
    # 计算卫星位置
    r = EARTH_RADIUS + height
    
    # 首先在 xz 平面计算位置
    x = r * math.cos(pos_rad)
    z = r * math.sin(pos_rad)
    
    # 根据轨道倾角旋转
    y = z * math.sin(incl_rad)
    z = z * math.cos(incl_rad)
    
    # 根据轨道平面旋转
    final_x = x * math.cos(orbit_rad) - y * math.sin(orbit_rad)
    final_y = x * math.sin(orbit_rad) + y * math.cos(orbit_rad)
    final_z = z
    
    return final_x, final_y, final_z

def create_satellite_topology_3d():
    G = nx.Graph()
    
    # 添加 MEO 节点和位置
    meo_positions = {}
    for orbit in range(NUM_ORBITS_MEO):
        for pos in range(SATS_PER_ORBIT_MEO):
            meo_name = f'meo{orbit * SATS_PER_ORBIT_MEO + pos + 1}'
            G.add_node(meo_name, node_type='meo')
            
            # 计算 MEO 位置
            x, y, z = calculate_satellite_position(
                orbit, pos, ORBIT_HEIGHT_MEO, INCLINATION, 
                NUM_ORBITS_MEO, SATS_PER_ORBIT_MEO
            )
            meo_positions[meo_name] = (x, y, z)
    
    # 添加 LEO 节点和位置
    leo_positions = {}
    for orbit in range(NUM_ORBITS_LEO):
        for pos in range(SATS_PER_ORBIT_LEO):
            leo_name = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 1}'
            G.add_node(leo_name, node_type='leo')
            
            # 计算 LEO 位置
            x, y, z = calculate_satellite_position(
                orbit, pos, ORBIT_HEIGHT_LEO, INCLINATION,
                NUM_ORBITS_LEO, SATS_PER_ORBIT_LEO
            )
            leo_positions[leo_name] = (x, y, z)
            
            # 连接同一轨道上的相邻卫星
            if pos > 0:
                prev_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos}'
                G.add_edge(leo_name, prev_leo, link_type='intra_orbit')
            if pos == SATS_PER_ORBIT_LEO - 1:  # 连接轨道首尾
                first_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + 1}'
                G.add_edge(leo_name, first_leo, link_type='intra_orbit')
            
            # 连接相邻轨道的卫星
            if orbit > 0:
                left_leo = f'leo{(orbit-1) * SATS_PER_ORBIT_LEO + pos + 1}'
                G.add_edge(leo_name, left_leo, link_type='inter_orbit')
            if orbit == NUM_ORBITS_LEO - 1:  # 连接首尾轨道
                first_orbit_leo = f'leo{pos + 1}'
                G.add_edge(leo_name, first_orbit_leo, link_type='inter_orbit')
    
    # 添加MEO-LEO连接（基于视线范围）
    for meo_name, meo_pos in meo_positions.items():
        for leo_name, leo_pos in leo_positions.items():
            # 计算MEO和LEO之间的距离
            distance = np.sqrt(sum((np.array(meo_pos) - np.array(leo_pos))**2))
            # 如果在可见范围内（这里使用一个简单的距离阈值）
            if distance < (ORBIT_HEIGHT_MEO - ORBIT_HEIGHT_LEO) * 1.5:
                G.add_edge(meo_name, leo_name, link_type='meo_leo')
    
    return G, meo_positions, leo_positions

def draw_earth_wireframe(ax, radius=EARTH_RADIUS, color='gray', alpha=0.2):
    """绘制地球线框"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def visualize_topology_3d():
    G, meo_positions, leo_positions = create_satellite_topology_3d()
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地球
    draw_earth_wireframe(ax)
    
    # 为LEO轨道面创建蓝色系的颜色方案（避开红色系）
    leo_colors = plt.cm.Blues(np.linspace(0.3, 0.9, NUM_ORBITS_LEO))
    
    # 分轨道面绘制LEO节点和连接
    for orbit in range(NUM_ORBITS_LEO):
        orbit_color = leo_colors[orbit]
        
        # 获取该轨道面的LEO卫星
        orbit_leos = [f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 1}' 
                     for pos in range(SATS_PER_ORBIT_LEO)]
        
        # 绘制该轨道面的LEO节点
        leo_x = [leo_positions[leo][0] for leo in orbit_leos]
        leo_y = [leo_positions[leo][1] for leo in orbit_leos]
        leo_z = [leo_positions[leo][2] for leo in orbit_leos]
        ax.scatter(leo_x, leo_y, leo_z, c=[orbit_color], s=50, 
                  label=f'LEO Orbit {orbit+1}')
        
        # 绘制轨道内连接（同一轨道面内的连接）
        for i in range(len(orbit_leos)):
            leo1 = orbit_leos[i]
            leo2 = orbit_leos[(i + 1) % len(orbit_leos)]
            x1, y1, z1 = leo_positions[leo1]
            x2, y2, z2 = leo_positions[leo2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                   color=orbit_color, alpha=0.5, linewidth=1)
        
        # 绘制轨道间连接
        if orbit < NUM_ORBITS_LEO - 1:
            for pos in range(SATS_PER_ORBIT_LEO):
                leo1 = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 1}'
                leo2 = f'leo{(orbit + 1) * SATS_PER_ORBIT_LEO + pos + 1}'
                x1, y1, z1 = leo_positions[leo1]
                x2, y2, z2 = leo_positions[leo2]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 
                       'gray', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # MEO使用醒目的红色系
    meo_x = [pos[0] for pos in meo_positions.values()]
    meo_y = [pos[1] for pos in meo_positions.values()]
    meo_z = [pos[2] for pos in meo_positions.values()]
    ax.scatter(meo_x, meo_y, meo_z, c='darkred', s=100, label='MEO')
    
    # MEO-LEO连接使用橙色
    for edge in G.edges(data=True):
        if edge[2]['link_type'] == 'meo_leo':
            if edge[0].startswith('meo'):
                x1, y1, z1 = meo_positions[edge[0]]
                x2, y2, z2 = leo_positions[edge[1]]
            else:
                x1, y1, z1 = leo_positions[edge[0]]
                x2, y2, z2 = meo_positions[edge[1]]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='orange', alpha=0.1)
    
    # 设置图形属性
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Satellite Network Topology')
    
    # 设置坐标轴比例相等
    max_range = np.array([
        max(leo_x + meo_x) - min(leo_x + meo_x),
        max(leo_y + meo_y) - min(leo_y + meo_y),
        max(leo_z + meo_z) - min(leo_z + meo_z)
    ]).max() / 2.0
    
    mid_x = (max(leo_x + meo_x) + min(leo_x + meo_x)) * 0.5
    mid_y = (max(leo_y + meo_y) + min(leo_y + meo_y)) * 0.5
    mid_z = (max(leo_z + meo_z) + min(leo_z + meo_z)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    visualize_topology_3d()
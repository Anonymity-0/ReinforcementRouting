from agi.stk12 import *
from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *
from agi.stk12.vgt import *
from datetime import datetime, timedelta
import os

def create_hierarchical_constellation():
    try:
        # 尝试连接到STK12，如果失败则创建新实例
        print("正在连接STK12...")
        try:
            stk = STKDesktop.AttachToApplication()
        except:
            print("未找到运行中的STK12，正在启动新实例...")
            stk = STKDesktop.StartApplication(visible=True)
            stk.UserControl = True  # 允许用户控制
            stk.Visible = True      # 确保窗口可见
        
        root = stk.Root
        print("成功获取STK根对象")
        
        # 设置日期格式
        root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
        
        # 创建新场景
        root.NewScenario('Hierarchical_Constellation')
        scenario = root.CurrentScenario
        print("成功创建新场景")
        
        # 设置时间周期为当前时间开始的24小时
        current_time = datetime.utcnow()
        start_time = current_time.strftime("%d %b %Y %H:%M:%S.00")
        end_time = (current_time + timedelta(days=1)).strftime("%d %b %Y %H:%M:%S.00")
        
        print(f"设置仿真时间: {start_time} 到 {end_time}")
        scenario.SetTimePeriod(start_time, end_time)

        try:
            # 创建LEO种子卫星
            print("正在创建LEO种子卫星...")
            leo_seed = scenario.Children.New(AgESTKObjectType.eSatellite, 'LEO_Seed')
            
            # 设置LEO轨道参数
            keplerian = leo_seed.Propagator.InitialState.Representation.ConvertTo(
                AgEOrbitStateType.eOrbitStateClassical)
            
            # 设置轨道形状类型和位置类型
            keplerian.SizeShapeType = 4  # Semimajor Axis and Eccentricity
            keplerian.LocationType = 5  # eLocationTrueAnomaly
            keplerian.Orientation.AscNodeType = 1  # eAscNodeRAAN
            
            # 设置LEO轨道参数
            keplerian.SizeShape.SemiMajorAxis = 7871  # km (1500km + 地球半径)
            keplerian.SizeShape.Eccentricity = 0
            keplerian.Orientation.Inclination = 55  # deg
            keplerian.Orientation.ArgOfPerigee = 0  # deg
            keplerian.Orientation.AscNode.Value = 0  # deg
            keplerian.Location.Value = 0  # deg True anomaly
            
            # 应用轨道参数并传播
            leo_seed.Propagator.InitialState.Representation.Assign(keplerian)
            leo_seed.Propagator.Propagate()
            print("成功设置LEO轨道参数")
            
            # 创建LEO Walker星座
            cmd = ('Walker */Satellite/LEO_Seed Type Delta NumPlanes ' + 
                  str(int(16)) + 
                  ' NumSatsPerPlane ' + 
                  str(int(16)) + 
                  ' InterPlanePhaseIncrement ' + 
                  str(int(0)) + 
                  ' ColorByPlane Yes' + 
                  ' ConstellationName LEO_Walker')
            root.ExecuteCommand(cmd)
            print("成功创建LEO Walker星座")
            
            # 卸载LEO种子卫星
            leo_seed.Unload()

            # 创建MEO种子卫星
            print("正在创建MEO种子卫星...")
            meo_seed = scenario.Children.New(AgESTKObjectType.eSatellite, 'MEO_Seed')
            
            # 设置MEO轨道参数
            keplerian = meo_seed.Propagator.InitialState.Representation.ConvertTo(
                AgEOrbitStateType.eOrbitStateClassical)
            
            keplerian.SizeShapeType = 4
            keplerian.LocationType = 5
            keplerian.Orientation.AscNodeType = 1
            
            # 设置MEO轨道参数 (高度设为8000km以覆盖LEO星座)
            keplerian.SizeShape.SemiMajorAxis = 14371  # km (8000km + 地球半径)
            keplerian.SizeShape.Eccentricity = 0
            keplerian.Orientation.Inclination = 55  # 与LEO相同的倾角以保持覆盖
            keplerian.Orientation.ArgOfPerigee = 0
            keplerian.Orientation.AscNode.Value = 0
            keplerian.Location.Value = 0
            
            meo_seed.Propagator.InitialState.Representation.Assign(keplerian)
            meo_seed.Propagator.Propagate()
            print("成功设置MEO轨道参数")
            
            # 创建MEO Walker星座 (2轨道面，每面8颗卫星)
            cmd = ('Walker */Satellite/MEO_Seed Type Delta NumPlanes ' + 
                  str(int(2)) + 
                  ' NumSatsPerPlane ' + 
                  str(int(8)) + 
                  ' InterPlanePhaseIncrement ' + 
                  str(int(0)) + 
                  ' ColorByPlane Yes' + 
                  ' ConstellationName MEO_Walker')
            root.ExecuteCommand(cmd)
            print("成功创建MEO Walker星座")
            
            # 卸载MEO种子卫星
            meo_seed.Unload()
            
            print("正在创建卫星间连接...")
            
            # 获取所有MEO和LEO卫星
            meo_const = scenario.Children.Item("MEO_Walker")  # 获取MEO星座对象
            leo_const = scenario.Children.Item("LEO_Walker")  # 获取LEO星座对象
            
            meo_sats = []
            leo_sats = []
            
            # 获取星座中的所有卫星
            for i in range(meo_const.Objects.Count):
                meo_sats.append(meo_const.Objects.Item(i))
            
            for i in range(leo_const.Objects.Count):
                leo_sats.append(leo_const.Objects.Item(i))
            
            # 打印卫星名称以进行调试
            print("MEO卫星列表:")
            for sat in meo_sats:
                print(f"- {sat.InstanceName}")
            
            # 为每个MEO卫星添加传感器并创建与其覆盖范围内LEO卫星的连接
            for meo_sat in meo_sats:
                try:
                    # 添加传感器
                    sensor_name = f"Sensor_{meo_sat.InstanceName.replace('/', '_')}"  # 替换可能的非法字符
                    print(f"正在为{meo_sat.InstanceName}创建传感器: {sensor_name}")
                    sensor = meo_sat.Children.New(AgESTKObjectType.eSensor, sensor_name)
                    
                    # 设置传感器为简单圆锥形
                    sensor.Pattern.ConeAngle = 60  # 60度锥角
                    sensor.Pattern.Type = "Simple Conic"
                    
                    # 为每个LEO卫星计算可见性
                    for leo_sat in leo_sats:
                        access = sensor.GetAccessToObject(leo_sat)
                        access.ComputeAccess()
                        
                        # 如果在仿真期间有可见性，则创建通信链路
                        if access.ComputedAccessIntervalTimes.Count > 0:
                            comm_name = f"Comm_{meo_sat.InstanceName}_{leo_sat.InstanceName}"
                            comm = scenario.Children.New(AgESTKObjectType.eCommSystem, comm_name)
                            comm.SetTransmitter(meo_sat)
                            comm.SetReceiver(leo_sat)
                        
                        # 删除Access对象以释放内存
                        access.RemoveAccess()
                except Exception as e:
                    print(f"为卫星 {meo_sat.InstanceName} 创建传感器时出错: {str(e)}")

            # LEO卫星之间的网格连接保持不变
            num_planes = 16
            sats_per_plane = 16
            
            for i in range(len(leo_sats)):
                current_plane = i // sats_per_plane
                current_pos = i % sats_per_plane
                
                # 连接同一轨道面的上下卫星
                up_idx = i - 1 if current_pos > 0 else i + sats_per_plane - 1
                down_idx = i + 1 if current_pos < sats_per_plane - 1 else i - sats_per_plane + 1
                
                # 连接左右轨道面的对应卫星
                left_plane = (current_plane - 1) % num_planes
                right_plane = (current_plane + 1) % num_planes
                left_idx = left_plane * sats_per_plane + current_pos
                right_idx = right_plane * sats_per_plane + current_pos
                
                # 创建连接
                for neighbor_idx in [up_idx, down_idx, left_idx, right_idx]:
                    comm_name = f"Comm_{leo_sats[i].InstanceName}_{leo_sats[neighbor_idx].InstanceName}"
                    comm = scenario.Children.New(AgESTKObjectType.eCommSystem, comm_name)
                    comm.SetTransmitter(leo_sats[i])
                    comm.SetReceiver(leo_sats[neighbor_idx])
            
            print("成功创建卫星间连接")

        except Exception as e:
            print(f"星座创建失败: {str(e)}")
        
        print("\nSTK场景已创建完成。按Enter键退出...")
        input()
        
    except Exception as e:
        print(f"创建失败: {str(e)}")

if __name__ == "__main__":
    create_hierarchical_constellation()
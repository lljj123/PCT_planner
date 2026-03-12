#!/usr/bin/env python3
import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Path
import numpy as np

class PathVisualizer:
    def __init__(self):
        # 初始化节点
        rospy.init_node('path_visualizer_matplotlib', anonymous=True)
        
        # 订阅路径话题
        self.path_sub = rospy.Subscriber("/pct_path", Path, self.path_callback)
        
        # 存储路径数据
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.new_data_available = False

        # 初始化 Matplotlib 图形
        plt.ion()  # 开启交互模式，允许动态更新
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.setup_plot()
        
        print("Waiting for path data on topic /pct_path ...")

    def setup_plot(self):
        """设置图形的基本属性"""
        self.ax.set_title("Real-time Path Visualization")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.grid(True)

    def path_callback(self, msg):
        """ROS 回调函数：解析 Path 消息"""
        temp_x = []
        temp_y = []
        temp_z = []

        # 遍历路径中的所有位姿点
        for pose_stamped in msg.poses:
            point = pose_stamped.pose.position
            temp_x.append(point.x)
            temp_y.append(point.y)
            temp_z.append(point.z)

        # 更新数据
        self.x_data = temp_x
        self.y_data = temp_y
        self.z_data = temp_z
        self.new_data_available = True
        
        # 打印简要信息
        rospy.loginfo(f"Received path with {len(self.x_data)} points.")

    def update_plot(self):
        """主循环：更新绘图"""
        if self.new_data_available:
            # 清除并重绘，保持视角
            self.ax.cla()
            self.setup_plot()
            
            # 绘制路径线
            self.ax.plot(self.x_data, self.y_data, self.z_data, 
                        label='Planned Path', color='blue', linewidth=2, marker='.', markersize=4)
            
            # 标记起点（绿色）和终点（红色）
            if len(self.x_data) > 0:
                self.ax.scatter(self.x_data[0], self.y_data[0], self.z_data[0], 
                              color='green', s=100, label='Start', marker='o')
                self.ax.scatter(self.x_data[-1], self.y_data[-1], self.z_data[-1], 
                              color='red', s=100, label='Goal', marker='x')

            self.ax.legend()
            
            # 自动调整坐标轴范围以适应路径（可选，如果不喜欢跳动可以注释掉）
            # self.set_equal_aspect_ratio() 

            self.new_data_available = False
            plt.draw()

        # 处理 GUI 事件，保持窗口响应
        plt.pause(0.1)

    def set_equal_aspect_ratio(self):
        """辅助函数：尝试保持 3D 轴比例一致"""
        # 注意：Matplotlib 3D 的 equal aspect ratio 支持不如 2D 完善，这是一个简单的近似
        if not self.x_data: return
        max_range = np.array([max(self.x_data)-min(self.x_data), 
                              max(self.y_data)-min(self.y_data), 
                              max(self.z_data)-min(self.z_data)]).max() / 2.0
        mid_x = (max(self.x_data)+min(self.x_data)) * 0.5
        mid_y = (max(self.y_data)+min(self.y_data)) * 0.5
        mid_z = (max(self.z_data)+min(self.z_data)) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            try:
                self.update_plot()
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        plt.close(self.fig)

if __name__ == '__main__':
    try:
        viz = PathVisualizer()
        viz.run()
    except rospy.ROSInterruptException:
        pass
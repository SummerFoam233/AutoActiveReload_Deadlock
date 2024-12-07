import sys
import keyboard
import time
import numpy as np
import random
import threading
import json
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QComboBox,
    QCheckBox, QLabel, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from mss import mss
import cv2
import os


# 配置文件路径
CONFIG_FILE = "config.json"

# 定义分辨率参数
RESOLUTIONS = {
    "4K": {
        "side_length": 205,
        "inner_radius": 82,
        "outer_radius": 100,
        "red_pixels": 4656,
        "green_pixels": 1002
    },
    "2K_16_10": {
        "side_length": 155,
        "inner_radius": 60,
        "outer_radius": 75,
        "red_pixels": 2878,
        "green_pixels": 619
    },
    "2K_16_9": {
        "side_length": 140,
        "inner_radius": 53,
        "outer_radius": 68,
        "red_pixels": 2571,
        "green_pixels": 553
    },
    "1080P": {
        "side_length": 103,  # 使用4K参数
        "inner_radius": 41,
        "outer_radius": 50,
        "red_pixels": 2328,
        "green_pixels": 501
    }
}

# 定义需要检测的颜色和容差
RED_COLOR = np.array([178, 28, 28])
GREEN_COLOR = np.array([187, 223, 162])
BRIGHT_GREEN_COLOR = np.array([123, 241, 109])
TOLERANCE = 20  # 容差
RED_TOLERANCE = 50  # 容差
RED_PIXCEL_THRESHOLD = 4500  # 红色像素点阈值，用于判断QTE

# 全局变量
block_r = False
is_key_blocked = False
auto_mode = False
semi_auto_mode = False
pre_input_mode = False
manual_mode = False
debug_mode = False
stop_flag = False
state_lock = threading.Lock()
current_resolution = "4K"
qte_last_detected_time = 0

# 创建配置文件，如果不存在则创建默认配置
def load_config():
    global debug_mode, current_resolution, auto_mode, semi_auto_mode, pre_input_mode, manual_mode
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            auto_mode = config.get("auto_mode", False)
            debug_mode = config.get("debug_mode", False)
            current_resolution = config.get("current_resolution", "4K")
            semi_auto_mode = config.get("semi_auto_mode", False)
            pre_input_mode = config.get("pre_input_mode", False)
            manual_mode = config.get("manual_mode", False)
    else:
        config = {
            "auto_mode": True,
            "debug_mode": False,
            "current_resolution": "4K",
            "semi_auto_mode": False,
            "pre_input_mode": False,
            "manual_mode": False,
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        current_resolution = "4K"
        auto_mode = True
        semi_auto_mode = False
        pre_input_mode = False
        manual_mode = False

def save_config():
    config = {
        "auto_mode": auto_mode,
        "debug_mode": debug_mode,
        "current_resolution": current_resolution,
        "semi_auto_mode": semi_auto_mode,
        "pre_input_mode": pre_input_mode,
        "manual_mode": manual_mode,
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

class DetectionWorker(QObject):
    # 定义信号
    update_image = pyqtSignal(QImage)
    update_status = pyqtSignal(bool)

    def __init__(self, region, ring_params):
        super().__init__()
        self.region = region
        self.ring_params = ring_params
        self.bright_green_counts = deque(maxlen=5)  # 存储亮绿色像素点数量
        self.block_r = False  # R 键阻止状态
        self.is_key_blocked = False  # 本地维护的 R 键阻止状态
        self.pre_input_r = False # 预输入状态
        self.semi_auto_active = False  # 半自动模式激活标志
        self.semi_auto_ignore_r = False
        self.delay_min = 0.01
        self.delay_max = 0.05
        
    def update_parameters(self, region, ring_params):
        self.region = region
        self.ring_params = ring_params


    def run(self):
        global stop_flag, auto_mode, debug_mode
        sct = mss()

        while not stop_flag:
            start_time = time.time()
            sct_img = sct.grab(self.region)
            img = np.array(sct_img)[:, :, :3]  # 去掉 alpha 通道
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 创建圆环掩码（包含角度限制）
            center_x = self.ring_params['center_x']
            center_y = self.ring_params['center_y']
            inner_radius = self.ring_params['inner_radius']
            outer_radius = self.ring_params['outer_radius']

            Y, X = np.ogrid[:self.region['height'], :self.region['width']]
            dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            angle = np.degrees(np.arctan2(Y - center_y, X - center_x))
            angle = (angle + 360) % 360  # 将角度转换到 [0, 360)
            
            red_angle_mask = (angle >= 107.5) & (angle <= 270)  # 红色区域角度范围
            red_ring_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius) & red_angle_mask

          
            # 绿色检测区域
            green_angle_mask = (angle >= 72.5) & (angle <= 107.5)
            green_ring_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius) & green_angle_mask
            
            # 转换到 HSV
            hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            # 动态调整 HSV 范围
            avg_brightness = np.mean(hsv_img[:, :, 2])  # V 通道均值
            # 基于截图范围的综合 HSV 范围
            if avg_brightness > 200:  # 高亮场景
                lower_red1 = np.array([69, 0, 0])  # 综合低色相的下限
                upper_red1 = np.array([179, 255, 255])  # 综合低色相的上限
                lower_red2 = np.array([0, 0, 0]) # 避免引用错误
                upper_red2 = np.array([0, 0, 0])
            else:  # 正常亮度场景
                lower_red1 = np.array([0, 49, 0])  # 综合较低亮度下的 HSV 范围
                upper_red1 = np.array([39, 255, 255])  # 高饱和度的红色范围
                lower_red2 = np.array([123, 49, 17])  # 高色相部分的下限
                upper_red2 = np.array([179, 255, 255])  # 高色相部分的上限
            
            # 创建红色掩码
            mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            mask_red_full = cv2.bitwise_or(mask_red1, mask_red2)
            
            # 转换到 Lab
            lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
            lower_red_lab = np.array([20, 140, 80])
            upper_red_lab = np.array([255, 200, 120])
            mask_red_lab = cv2.inRange(lab_img, lower_red_lab, upper_red_lab)
            
            # 合并 HSV 和 Lab 掩码
            final_red_mask = cv2.bitwise_or(mask_red_full, mask_red_lab)
            
            # 应用到红色检测区域
            red_mask_filtered = np.zeros_like(final_red_mask, dtype=np.uint8)
            red_mask_filtered[red_ring_mask] = final_red_mask[red_ring_mask]
            
            # 统计 HSV 和 Lab 红色像素数量
            red_pixel_count = np.sum(red_mask_filtered > 0)
            
            # 加载掩膜图像
            red_pixel_mask = cv2.imread("red_pixel_mask.png", cv2.IMREAD_GRAYSCALE)
            if red_pixel_mask is None:
                raise FileNotFoundError("掩膜图像 red_pixel_mask.png 未找到")
            
            # 确保掩膜图像尺寸为 40x40
            if red_pixel_mask.shape != (40, 40):
                raise ValueError(f"掩膜图像尺寸不正确，当前尺寸为 {red_pixel_mask.shape}，需要为 (40, 40)")
            
            # 中心区域 40x40 截图
            height, width = img_rgb.shape[:2]
            center_x, center_y = width // 2, height // 2
            half_box = 20  # 正方形边长的一半
            center_region_bgr = img_rgb[center_y - half_box:center_y + half_box, center_x - half_box:center_x + half_box]
            center_region = cv2.cvtColor(center_region_bgr, cv2.COLOR_BGR2RGB)
            
            # 检测中心区域纯红色像素点
            red_tolerance = 10
            
            # 分解图片的 B, G, R 通道
            blue_channel, green_channel, red_channel = cv2.split(center_region)
            
            # 检测红色像素点
            red_condition = (red_channel >= 255 - red_tolerance)
            blue_condition = (blue_channel <= red_tolerance)
            green_condition = (green_channel <= red_tolerance)
            
            # 生成满足条件的掩膜
            center_red_mask = red_condition & blue_condition & green_condition
            
            # 将掩膜应用到 red_pixel_mask 中
            # 注意，red_pixel_mask 的非零区域是掩膜允许的区域
            refined_red_mask = center_red_mask & (red_pixel_mask > 0)
            
            # 统计满足掩膜条件的红色像素点数量
            refined_red_count = np.sum(refined_red_mask)
            
            # 判断 QTE 是否激活
            is_qte_active = refined_red_count > 300

            # 绿色 HSV 范围
            lower_green = np.array([35, 40, 40])  # 调整后的绿色色相范围
            upper_green = np.array([85, 255, 255])
            
            # 创建绿色掩码
            mask_green_full = cv2.inRange(hsv_img, lower_green, upper_green)
            
            # 剔除红色影响：仅保留非红色的绿色像素
            mask_green_filtered_no_red = cv2.bitwise_and(mask_green_full, cv2.bitwise_not(mask_red_full))
  
            # 应用绿色掩码到圆环检测区域
            green_mask_filtered = np.zeros_like(mask_green_filtered_no_red, dtype=np.uint8)
            green_mask_filtered[green_ring_mask] = mask_green_filtered_no_red[green_ring_mask]
            
            # 统计绿色像素数量
            green_pixel_count = np.sum(green_mask_filtered > 0)
            has_green = green_pixel_count > 801
            
            # 更新绿色像素点数量队列
            self.bright_green_counts.append(green_pixel_count)
            
            # 检查绿色像素点数量是否呈线性减少
            qte_detected = False
            if len(self.bright_green_counts) == self.bright_green_counts.maxlen:
                counts = list(self.bright_green_counts)
                x = np.arange(len(counts))
                y = np.array(counts)
                A = np.vstack([x, np.ones(len(x))]).T
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                if slope < -25:  # 斜率阈值，可根据需要调整
                    qte_detected = True
            
            # 更新 block_r
            if is_qte_active:
                if qte_detected:
                    self.block_r = False
                else:
                    self.block_r = True
            else:
                self.block_r = False

            # # 按键控制逻辑
            # if self.block_r:
            #     if not self.is_key_blocked:
            #         # 阻止 R 键输入
            #         keyboard.block_key('r')
            #         self.is_key_blocked = True
            #         self.r_pre_input_active = False  # 初始化预输入状态
            #         print(f"[{time.strftime('%H:%M:%S')}] 已阻止 R 键输入")
            #     else:
            #         # 检测用户是否按住 R 键，启用预输入
            #         if keyboard.is_pressed('r'):
            #             self.r_pre_input_active = True
            #             print(f"[{time.strftime('%H:%M:%S')}] 用户按住 R，预输入激活")
            #         else:
            #             # 用户抬起 R 键，取消预输入
            #             self.r_pre_input_active = False
            
            # else:
            #     if self.is_key_blocked:
            #         # 释放 R 键
            #         keyboard.unblock_key('r')
            #         keyboard.release('r')
            #         self.is_key_blocked = False
            #         print(f"[{time.strftime('%H:%M:%S')}] 已释放 R 键")
            
            #         # 如果预输入激活，触发按下 R
            #         if self.r_pre_input_active:
            #             delay = random.uniform(0.01, 0.05)  # 随机延迟 0.1 ~ 0.3 秒
            #             time.sleep(delay)
            #             keyboard.press_and_release('r')
            #             self.r_pre_input_active = False  # 重置预输入状态
            #             print(f"[{time.strftime('%H:%M:%S')}] 触发 R 的预输入操作，延迟 {delay:.3f} 秒")
            
            #         # 自动装填逻辑
            #         if auto_mode and qte_detected:
            #             delay = random.uniform(0.01, 0.05)  # 随机延迟 0.1 ~ 0.3 秒
            #             time.sleep(delay)
            #             keyboard.press_and_release('r')
            #             print(f"[{time.strftime('%H:%M:%S')}] 自动装填：按下 R 键，延迟 {delay:.3f} 秒")

            # 按键控制逻辑
            # 按键控制逻辑
            if self.block_r:
                if not self.is_key_blocked:
                    # 阻止 R 键输入
                    keyboard.block_key('r')
                    self.is_key_blocked = True
                    self.r_pre_input_active = False  # 初始化预输入状态
                    self.last_r_pressed = False  # 初始化最近 R 按键状态
                    print(f"[{time.strftime('%H:%M:%S')}] 已阻止 R 键输入")
                else:
                    # 当半自动模式开启时，如果按下 R 键，则关闭模式
                    if semi_auto_mode and keyboard.is_pressed('r'):
                        if self.semi_auto_active:
                            self.semi_auto_active = False
                            print(f"[{time.strftime('%H:%M:%S')}] 半自动关闭")
                        else:
                            self.semi_auto_active = True
                            print(f"[{time.strftime('%H:%M:%S')}] 半自动开启")
            
                    # 检查是否按住 R 键，进入预输入状态
                    if pre_input_mode:
                        if keyboard.is_pressed('r'):
                            self.r_pre_input_active = True
                            print(f"[{time.strftime('%H:%M:%S')}] 用户按住 R，预输入激活")
                        else:
                            self.r_pre_input_active = False
            
            else:
                # 当 block_r 为 False 时，释放 R 键
                if self.is_key_blocked:
                    keyboard.unblock_key('r')
                    keyboard.release('r')
                    self.is_key_blocked = False
                    print(f"[{time.strftime('%H:%M:%S')}] 已释放 R 键")
            
                    # 根据模式执行释放后的逻辑
                    if auto_mode and qte_detected:
                        # 自动模式下立即触发 R
                        delay = random.uniform(self.delay_min, self.delay_max)
                        time.sleep(delay)
                        keyboard.press_and_release('r')
                        print(f"[{time.strftime('%H:%M:%S')}] 自动模式：按下 R 键，延迟 {delay:.3f} 秒")
            
                    elif pre_input_mode and self.r_pre_input_active:
                        # 预输入模式触发 R
                        delay = random.uniform(self.delay_min, self.delay_max)
                        time.sleep(delay)
                        keyboard.press_and_release('r')
                        self.r_pre_input_active = False  # 重置预输入状态
                        print(f"[{time.strftime('%H:%M:%S')}] 触发 R 的预输入操作，延迟 {delay:.3f} 秒")
            
                    elif semi_auto_mode and self.semi_auto_active and qte_detected:
                        # 半自动模式触发 R
                        delay = random.uniform(self.delay_min, self.delay_max)
                        time.sleep(delay)
                        keyboard.press_and_release('r')
                        self.semi_auto_active = False  # 触发一次后关闭半自动
                        print(f"[{time.strftime('%H:%M:%S')}] 半自动模式：按下 R 键，延迟 {delay:.3f} 秒")
            
                    elif manual_mode:
                        # 手动模式无特殊逻辑
                        print(f"[{time.strftime('%H:%M:%S')}] 手动模式：无需阻止 R 键")



            if debug_mode:
                display_image = img_rgb.copy()
            
                # 绘制检测区域
                cv2.ellipse(display_image, (center_x, center_y), (outer_radius, outer_radius), 0, 107.5, 270, (0, 0, 255), 1)  # 红色区域
                cv2.ellipse(display_image, (center_x, center_y), (inner_radius, inner_radius), 0, 107.5, 270, (0, 0, 255), 1)  # 红色区域
                cv2.ellipse(display_image, (center_x, center_y), (outer_radius, outer_radius), 0, 72.5, 107.5, (0, 255, 0), 1)  # 绿色区域
                cv2.ellipse(display_image, (center_x, center_y), (inner_radius, inner_radius), 0, 72.5, 107.5, (0, 255, 0), 1)  # 绿色区域
            
                # 显示状态文字
                r_block_text = f"R_Block: {self.block_r}"
                is_qte_active_text = f"QTE_Active: {is_qte_active}"
                qte_detected_text = f"QTE_Detected: {qte_detected}"
                
                # 调整文字的字号和位置
                font_scale = 0.5  # 字号稍小
                text_color = (0,255,255)  # 绿色字体
                thickness = 2  # 字体厚度
            
                cv2.putText(display_image, r_block_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                cv2.putText(display_image, is_qte_active_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                cv2.putText(display_image, qte_detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
                # 在中心区域描绿检测出的红色像素点
                center_region_bgr = display_image[center_y - half_box:center_y + half_box, center_x - half_box:center_x + half_box]
                center_region_bgr[center_red_mask] = [0, 255, 0]  # 将检测到的红色像素点描为绿色
                display_image[center_y - half_box:center_y + half_box, center_x - half_box:center_x + half_box] = center_region_bgr
            
                # 转换为 QImage
                height, width, channel = display_image.shape
                bytes_per_line = 3 * width
                q_img = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.update_image.emit(q_img)

        elapsed_time = time.time() - start_time
        time.sleep(max(0, (1 / 60) - elapsed_time))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTE 自动装填检测")
        self.setWindowIcon(QIcon('ActiveReload.ico'))

        # 加载配置
        load_config()

        # 主窗口布局
        main_layout = QHBoxLayout()

        # 左侧控件布局
        control_layout = QVBoxLayout()

        # 分辨率下拉框
        self.resolution_label = QLabel("选择分辨率：")
        self.resolution_dropdown = QComboBox()
        self.resolution_dropdown.addItems(["4K", "2K_16_10", "2K_16_9", "1080P"])
        self.resolution_dropdown.setCurrentText(current_resolution)
        self.resolution_dropdown.currentTextChanged.connect(self.on_resolution_change)

        control_layout.addWidget(self.resolution_label)
        control_layout.addWidget(self.resolution_dropdown)

        # 自动模式 checkbox
        self.auto_mode_checkbox = QCheckBox("自动模式")
        self.auto_mode_checkbox.setChecked(auto_mode)
        self.auto_mode_checkbox.stateChanged.connect(lambda: self.set_mode("auto"))
        control_layout.addWidget(self.auto_mode_checkbox)

        # 半自动模式 checkbox
        self.semi_auto_mode_checkbox = QCheckBox("半自动模式")
        self.semi_auto_mode_checkbox.setChecked(semi_auto_mode)
        self.semi_auto_mode_checkbox.stateChanged.connect(lambda: self.set_mode("semi_auto"))
        control_layout.addWidget(self.semi_auto_mode_checkbox)

        # 预输入模式 checkbox
        self.pre_input_mode_checkbox = QCheckBox("预输入模式")
        self.pre_input_mode_checkbox.setChecked(pre_input_mode)
        self.pre_input_mode_checkbox.stateChanged.connect(lambda: self.set_mode("pre_input"))
        control_layout.addWidget(self.pre_input_mode_checkbox)

        # 手动模式 checkbox
        self.manual_mode_checkbox = QCheckBox("手动模式")
        self.manual_mode_checkbox.setChecked(manual_mode)
        self.manual_mode_checkbox.stateChanged.connect(lambda: self.set_mode("manual"))
        control_layout.addWidget(self.manual_mode_checkbox)

        # Debug checkbox
        self.debug_checkbox = QCheckBox("是否展示debug检测框")
        self.debug_checkbox.setChecked(debug_mode)
        self.debug_checkbox.stateChanged.connect(self.toggle_debug_mode)

        control_layout.addWidget(self.debug_checkbox)

        # R_Block 状态标签
        self.status_label = QLabel(f"R_Block: {block_r}")
        control_layout.addWidget(self.status_label)

        # 添加伸缩
        control_layout.addStretch()

        # 将左侧控件布局放入容器
        self.control_container = QWidget()
        self.control_container.setFixedWidth(300)  # 固定宽度
        self.control_container.setLayout(control_layout)

        # 将左侧控件容器添加到主布局
        main_layout.addWidget(self.control_container)

        # 右侧调试窗口
        self.display_label = QLabel()
        self.display_label.setFixedSize(640, 360)  # 固定调试窗口大小
        self.display_label.setAlignment(Qt.AlignCenter)

        # 调试窗口容器，固定宽度
        self.debug_container = QWidget()
        self.debug_container.setFixedWidth(640)  # 宽度与调试窗口一致
        debug_layout = QVBoxLayout()
        debug_layout.addWidget(self.display_label)
        self.debug_container.setLayout(debug_layout)

        # 将调试窗口容器添加到主布局
        main_layout.addWidget(self.debug_container, alignment=Qt.AlignRight)

        # 设置中心窗口
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 初始化截图区域和圆环参数
        self.update_detection_parameters(current_resolution)

        # 启动检测线程
        self.detection_worker = DetectionWorker(self.region, self.ring_params)
        self.detection_worker.update_image.connect(self.update_debug_image)
        self.detection_worker.update_status.connect(self.update_r_block_status)

        self.detection_thread = threading.Thread(target=self.detection_worker.run, daemon=True)
        self.detection_thread.start()

        # 设置窗口始终置顶
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)


    def set_mode(self, mode):
        global auto_mode, semi_auto_mode, pre_input_mode, manual_mode
    
        # 重置所有模式
        auto_mode = semi_auto_mode = pre_input_mode = manual_mode = False
    
        # 设置当前模式
        if mode == "auto":
            auto_mode = True
        elif mode == "semi_auto":
            semi_auto_mode = True
        elif mode == "pre_input":
            pre_input_mode = True
        elif mode == "manual":
            manual_mode = True
    
        # 更新 Checkbox 状态，确保互斥
        # 临时断开信号，避免重复触发
        self.auto_mode_checkbox.blockSignals(True)
        self.auto_mode_checkbox.setChecked(auto_mode)
        self.auto_mode_checkbox.blockSignals(False)
    
        self.semi_auto_mode_checkbox.blockSignals(True)
        self.semi_auto_mode_checkbox.setChecked(semi_auto_mode)
        self.semi_auto_mode_checkbox.blockSignals(False)
    
        self.pre_input_mode_checkbox.blockSignals(True)
        self.pre_input_mode_checkbox.setChecked(pre_input_mode)
        self.pre_input_mode_checkbox.blockSignals(False)
    
        self.manual_mode_checkbox.blockSignals(True)
        self.manual_mode_checkbox.setChecked(manual_mode)
        self.manual_mode_checkbox.blockSignals(False)
    
        # 保存配置
        save_config()

        
    def set_fixed_window_size(self, resolution):
        # 动态计算窗口大小，根据控件和调试窗口的大小
        control_width = 300
        debug_width = 640
        spacing = 50
        window_width = control_width + debug_width + spacing

        window_height = 400
        self.setFixedSize(window_width, window_height)

    def update_detection_parameters(self, resolution):
        # 获取屏幕分辨率
        with mss() as sct:
            monitor = sct.monitors[1]
            screen_width = monitor['width']
            screen_height = monitor['height']

        params = RESOLUTIONS[resolution]
        side_length = params['side_length']
        inner_radius = params['inner_radius']
        outer_radius = params['outer_radius']

        left = int((screen_width - side_length) / 2)
        top = int((screen_height - side_length) / 2)
        self.region = {
            "top": top,
            "left": left,
            "width": side_length,
            "height": side_length
        }

        center_x = int(side_length / 2)
        center_y = int(side_length / 2)

        self.ring_params = {
            "center_x": center_x,
            "center_y": center_y,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "side_length": side_length
        }

    def on_resolution_change(self, resolution):
        global current_resolution
        current_resolution = resolution
        self.set_fixed_window_size(resolution)
        self.update_detection_parameters(resolution)

        self.detection_worker.update_parameters(self.region, self.ring_params)
        save_config()

    def toggle_debug_mode(self, state):
        global debug_mode
        debug_mode = state == Qt.Checked
        if debug_mode:
            if self.latest_q_img:
                pixmap = QPixmap.fromImage(self.latest_q_img).scaled(
                    self.display_label.width(),
                    self.display_label.height(),
                    Qt.KeepAspectRatio
                )
                self.display_label.setPixmap(pixmap)
            self.display_label.show()
        else:
            self.display_label.hide()
        save_config()

    def update_debug_image(self, q_img):
        self.latest_q_img = q_img
        if debug_mode:
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.display_label.width(),
                self.display_label.height(),
                Qt.KeepAspectRatio
            )
            self.display_label.setPixmap(pixmap)

    def update_r_block_status(self, status):
        self.status_label.setText(f"R_Block: {status}")
        save_config()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, '退出确认',
            '确定要退出程序吗？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            global stop_flag
            stop_flag = True
            event.accept()
        else:
            event.ignore()


# 启动 PyQt GUI
def main():
    load_config()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

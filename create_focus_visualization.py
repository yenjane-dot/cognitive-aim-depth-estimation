#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建推理聚焦效果图的可视化脚本
将9个不同区域的聚焦预测结果合成为一个综合展示图
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_resize_image(image_path, target_size=(200, 200)):
    """加载并调整图像大小"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

def create_compact_focus_visualization():
    """创建三张图垂直布局的聚焦效果可视化"""
    demo_results_dir = Path('demo_results')
    
    # 选择三个关键位置：上方、中心、下方
    positions = {
        'top': ('2_top_prediction.png', 'Top Focus'),
        'center': ('2_center_prediction.png', 'Center Focus'),
        'bottom': ('2_bottom_prediction.png', 'Bottom Focus')
    }
    
    # 创建垂直布局，适合长方形原图
    fig = plt.figure(figsize=(8, 15))  # 高瘦的比例，适合垂直排列
    fig.patch.set_facecolor('white')
    
    # 使用GridSpec创建1x3垂直布局
    gs = GridSpec(3, 1, figure=fig, 
                  left=0.1, right=0.9, top=0.92, bottom=0.08,
                  hspace=0.25)
    
    # 位置映射：从上到下
    layout_order = ['top', 'center', 'bottom']
    
    for i, pos_name in enumerate(layout_order):
        ax = fig.add_subplot(gs[i, 0])
        
        # 加载对应的图像
        filename, english_label = positions[pos_name]
        img_path = demo_results_dir / filename
        
        # 保持原图比例，不强制调整为正方形
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # 如果图像不存在，创建占位图
            img = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        # 显示图像，保持原始宽高比
        ax.imshow(img, aspect='auto')
        
        # 设置英文标题
        ax.set_title(english_label, fontsize=16, fontweight='bold', 
                    pad=15, color='#2C3E50')
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加精美边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_edgecolor('#34495E')
            spine.set_alpha(0.9)
        
        # 为中心图像添加特殊标记
        if pos_name == 'center':
            for spine in ax.spines.values():
                spine.set_edgecolor('#E74C3C')
                spine.set_linewidth(3.5)
    
    # 添加主标题
    fig.suptitle('Cognitive-Aim: Spatial Attention Control Demonstration', 
                fontsize=18, fontweight='bold', y=0.97, color='#2C3E50')
    
    # 添加说明文字
    fig.text(0.5, 0.03, 
            'The model supports precise spatial guidance for different focus regions',
            ha='center', va='center', fontsize=12, style='italic', color='#7F8C8D')
    
    # 保存高质量图像
    output_path = 'figure3_focus_guidance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"三张图垂直布局已保存至: {output_path}")
    
    plt.close()
    return output_path

if __name__ == '__main__':
    print("创建推理聚焦效果图...")
    
    # 检查demo_results目录是否存在
    if not Path('demo_results').exists():
        print("错误: demo_results目录不存在")
        exit(1)
    
    # 创建论文用紧凑版图像
    print("\n创建论文用紧凑版图像...")
    compact_path = create_compact_focus_visualization()
    
    print(f"\n完成! 生成了论文版图像: {compact_path}")
    print("\n可以用此图像替换原论文中的图3。")
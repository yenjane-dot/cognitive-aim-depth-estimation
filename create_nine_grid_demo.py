#!/usr/bin/env python3
"""
创建九宫格空间引导效果展示图
使用demo.py生成的9个方向的推理结果进行合成
采用3行布局，每行3个长方形图像
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import argparse

def create_nine_grid_layout(demo_results_dir: str, output_path: str):
    """
    创建3行x3列布局的空间引导效果图
    
    Args:
        demo_results_dir: demo_results目录路径
        output_path: 输出图像路径
    """
    
    # 九个方向的顺序 (3行x3列布局)
    directions = [
        ['top-left', 'top', 'top-right'],
        ['left', 'center', 'right'], 
        ['bottom-left', 'bottom', 'bottom-right']
    ]
    
    # 创建图形 - 适应长方形图像的比例
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.01, wspace=0.05)
    
    # 设置整体标题
    fig.suptitle('Cognitive-Aim: Spatial Attention Control Demonstration', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    for row in range(3):
        for col in range(3):
            direction = directions[row][col]
            
            # 查找对应的预测图像
            prediction_file = f"2_{direction}_prediction.png"
            prediction_path = os.path.join(demo_results_dir, prediction_file)
            
            if not os.path.exists(prediction_path):
                print(f"Warning: {prediction_path} not found, skipping {direction}")
                continue
                
            # 加载预测图像
            pred_img = Image.open(prediction_path)
            
            # 创建子图
            ax = fig.add_subplot(gs[row, col])
            
            # 显示预测图像
            ax.imshow(pred_img)
            ax.set_title(f'{direction.replace("-", "-").title()} Focus', 
                        fontsize=12, fontweight='bold', pad=8)
            ax.axis('off')
            
            # 添加边框突出显示
            if direction == 'center':
                # 中心位置用红色边框突出
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                # 其他位置用蓝色边框
                for spine in ax.spines.values():
                    spine.set_edgecolor('blue')
                    spine.set_linewidth(2)
                    spine.set_visible(True)
    
    # 添加说明文字
    fig.text(0.5, 0.02, 
             'The model supports precise spatial guidance for different focus regions',
             ha='center', fontsize=11, style='italic')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ 九宫格空间引导效果图已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='创建九宫格空间引导效果展示图')
    parser.add_argument('--demo_results', type=str, default='demo_results',
                       help='demo_results目录路径')
    parser.add_argument('--output', type=str, default='figure3_nine_grid_spatial_guidance.png',
                       help='输出图像路径')
    
    args = parser.parse_args()
    
    # 验证demo_results目录
    if not os.path.exists(args.demo_results):
        raise FileNotFoundError(f"demo_results目录不存在: {args.demo_results}")
    
    # 创建九宫格图像
    create_nine_grid_layout(args.demo_results, args.output)
    
    print(f"\n🎯 九宫格空间引导效果图生成完成!")
    print(f"📁 输出文件: {args.output}")
    print(f"🔍 展示了9种不同的空间引导效果")

if __name__ == '__main__':
    main()
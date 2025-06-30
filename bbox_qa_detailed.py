"""
详细回答关于bbox的三个问题
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from screenshot_utils import get_click_coordinate_by_bbox

def question_1_answer():
    """
    问题1: bbox是转换前的也就是原始比例的位置吗？
    """
    print("=" * 60)
    print("问题1: bbox是转换前的原始比例位置吗？")
    print("=" * 60)
    
    print("答案：是的！bbox使用的是相对于原始图像尺寸的比例坐标。")
    print()
    
    print("详细解释：")
    print("1. 图像处理流程：")
    print("   原始图像 → 预处理（可能缩放） → YOLO检测 → bbox坐标归一化")
    print()
    
    print("2. 关键代码分析（来自utils.py第443行）：")
    print("   ```python")
    print("   w, h = image_source.size  # 获取原始图像尺寸")
    print("   xyxy, logits, phrases = predict_yolo(...)  # YOLO检测，返回像素坐标")
    print("   xyxy = xyxy / torch.Tensor([w, h, w, h])  # 归一化为比例坐标")
    print("   ```")
    print()
    
    print("3. 这意味着：")
    print("   - YOLO模型检测出的是像素坐标")
    print("   - 然后除以原始图像的宽高，转换为0-1之间的比例")
    print("   - bbox保存的是相对于原始图像的比例位置")
    print("   - 这样设计保证了跨分辨率的通用性")
    print()
    
    # 示例演示
    print("4. 示例演示：")
    print("   假设原始图像尺寸：1920x1080")
    print("   YOLO检测到按钮位置：像素坐标 [768, 918, 1152, 1026]")
    print("   归一化后的bbox：[768/1920, 918/1080, 1152/1920, 1026/1080]")
    print("                  = [0.4, 0.85, 0.6, 0.95]")
    print("   这个bbox可以适用于任何分辨率的设备！")

def question_2_answer():
    """
    问题2: bbox中四个位置在视觉上来看就是一个方形块吗？
    """
    print("\n" + "=" * 60)
    print("问题2: bbox在视觉上是一个方形块吗？")
    print("=" * 60)
    
    print("答案：是的！bbox就是一个矩形区域。")
    print()
    
    print("详细解释：")
    print("1. bbox格式：[x1, y1, x2, y2]")
    print("   - (x1, y1): 左上角坐标")
    print("   - (x2, y2): 右下角坐标")
    print("   - 四个值确定一个矩形区域")
    print()
    
    print("2. 视觉表示：")
    print("   ```")
    print("   (x1,y1) ┌─────────────┐")
    print("           │             │")
    print("           │  UI元素     │  <- 这就是bbox框住的区域")
    print("           │             │")
    print("           └─────────────┘ (x2,y2)")
    print("   ```")
    print()
    
    print("3. 在OmniParser的标注图像中：")
    print("   - 你看到的每个带编号的框框就是一个bbox")
    print("   - 比如'Home 1'外面的矩形框就是Home按钮的bbox")
    print("   - 框框的颜色和编号只是为了可视化，实际数据就是4个坐标值")

def question_3_answer():
    """
    问题3: click的位置也是方形block吗？
    """
    print("\n" + "=" * 60)
    print("问题3: click的位置也是方形block吗？")
    print("=" * 60)
    
    print("答案：不是！click位置是一个点，不是方形。")
    print()
    
    print("详细解释：")
    print("1. get_click_coordinate_by_bbox()函数的作用：")
    print("   - 输入：方形的bbox区域 [x1, y1, x2, y2]")
    print("   - 输出：单个点击坐标 (click_x, click_y)")
    print("   - 计算方法：取bbox的中心点")
    print()
    
    print("2. 中心点计算公式：")
    print("   ```python")
    print("   center_x = (x1 + x2) / 2")
    print("   center_y = (y1 + y2) / 2")
    print("   click_x = center_x * screen_width")
    print("   click_y = center_y * screen_height")
    print("   ```")
    print()
    
    print("3. 视觉对比：")
    print("   ```")
    print("   bbox（方形区域）:        点击位置（单点）:")
    print("   ┌─────────────┐         ┌─────────────┐")
    print("   │             │         │             │")
    print("   │  UI元素     │   -->   │      ●      │  <- 点击这里")
    print("   │             │         │             │")
    print("   └─────────────┘         └─────────────┘")
    print("   ```")
    print()
    
    print("4. 实际点击行为：")
    print("   - 就像用手指或鼠标在屏幕上点击一个精确的像素点")
    print("   - 不是在整个方形区域上点击")
    print("   - 选择中心点是因为通常UI元素的中心最安全、最有效")
    print()
    
    # 数值示例
    print("5. 数值示例：")
    bbox = [0.4, 0.85, 0.6, 0.95]
    click_x, click_y = get_click_coordinate_by_bbox(bbox, 1920, 1080)
    print(f"   bbox区域: {bbox}")
    print(f"   区域大小: {(0.6-0.4)*1920:.0f}x{(0.95-0.85)*1080:.0f} 像素")
    print(f"   点击坐标: ({click_x}, {click_y}) - 只是一个点！")

def create_visual_demo():
    """
    创建可视化演示图
    """
    print("\n" + "=" * 60)
    print("可视化演示")
    print("=" * 60)
    
    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1：bbox原理
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    
    # 绘制一个bbox
    bbox_rect = patches.Rectangle((0.3, 0.6), 0.4, 0.2, 
                                 linewidth=3, edgecolor='red', 
                                 facecolor='lightblue', alpha=0.5)
    ax1.add_patch(bbox_rect)
    ax1.text(0.5, 0.7, 'UI元素\nbbox=[0.3,0.6,0.7,0.8]', 
            ha='center', va='center', fontsize=10, weight='bold')
    ax1.text(0.3, 0.85, '(x1,y1)', ha='center', va='bottom', fontsize=9)
    ax1.text(0.7, 0.55, '(x2,y2)', ha='center', va='top', fontsize=9)
    ax1.set_title('1. bbox是矩形区域', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 图2：点击位置
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    
    # 绘制同样的bbox，但加上中心点
    bbox_rect2 = patches.Rectangle((0.3, 0.6), 0.4, 0.2, 
                                  linewidth=2, edgecolor='red', 
                                  facecolor='none', linestyle='--')
    ax2.add_patch(bbox_rect2)
    
    # 中心点
    center_x, center_y = 0.5, 0.7
    ax2.plot(center_x, center_y, 'ro', markersize=10, label='点击位置')
    ax2.text(center_x, center_y-0.08, f'点击点\n({center_x:.1f}, {center_y:.1f})', 
            ha='center', va='top', fontsize=10, weight='bold')
    ax2.set_title('2. 点击位置是中心点', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 图3：不同分辨率的对比
    ax3.set_xlim(0, 1920)
    ax3.set_ylim(0, 1080)
    ax3.invert_yaxis()  # y轴向下
    
    # 转换为像素坐标
    bbox = [0.3, 0.6, 0.7, 0.8]
    pixel_x1 = bbox[0] * 1920
    pixel_y1 = bbox[1] * 1080
    pixel_w = (bbox[2] - bbox[0]) * 1920
    pixel_h = (bbox[3] - bbox[1]) * 1080
    
    bbox_rect3 = patches.Rectangle((pixel_x1, pixel_y1), pixel_w, pixel_h, 
                                  linewidth=3, edgecolor='blue', 
                                  facecolor='lightgreen', alpha=0.5)
    ax3.add_patch(bbox_rect3)
    
    # 点击位置
    click_x = (bbox[0] + bbox[2]) / 2 * 1920
    click_y = (bbox[1] + bbox[3]) / 2 * 1080
    ax3.plot(click_x, click_y, 'ro', markersize=8)
    ax3.text(click_x, click_y-50, f'点击坐标\n({click_x:.0f}, {click_y:.0f})', 
            ha='center', va='top', fontsize=10, weight='bold')
    
    ax3.set_title('3. 1920x1080屏幕上的实际像素', fontsize=12, weight='bold')
    ax3.set_xlabel('像素 X')
    ax3.set_ylabel('像素 Y')
    
    plt.tight_layout()
    plt.savefig('bbox_explanation.png', dpi=150, bbox_inches='tight')
    print("可视化图表已保存为 'bbox_explanation.png'")
    print("图表展示了：")
    print("1. bbox的矩形区域概念")
    print("2. 点击位置是bbox的中心点")
    print("3. 在实际屏幕分辨率下的像素位置")

def summary():
    """
    总结
    """
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    print("1. bbox是原始比例位置")
    print("   ✓ 使用原始图像尺寸进行归一化")
    print("   ✓ 保存为0-1之间的比例坐标")
    print("   ✓ 具有跨设备通用性")
    print()
    
    print("2. bbox是矩形区域")
    print("   ✓ [x1,y1,x2,y2] 确定一个矩形")
    print("   ✓ 框住整个UI元素")
    print("   ✓ 在标注图上显示为方框")
    print()
    
    print("3. 点击位置是单个像素点")
    print("   ✓ 计算bbox的中心坐标")
    print("   ✓ 转换为屏幕上的精确像素位置")
    print("   ✓ 模拟单点触摸/点击操作")
    print()
    
    print("关键理解：")
    print("• bbox(矩形) → get_click_coordinate_by_bbox() → 点击坐标(点)")
    print("• 从'区域'到'点'的转换")
    print("• 这是UI自动化的核心机制")

if __name__ == "__main__":
    question_1_answer()
    question_2_answer() 
    question_3_answer()
    create_visual_demo()
    summary() 
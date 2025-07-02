import cv2
import numpy as np
import time
import random
import string
from typing import List, Tuple, Dict, Union
from PIL import Image, ImageGrab
import pyautogui

def get_random_string(length: int = 8) -> str:
    """生成随机字符串"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def get_current_time() -> str:
    """获取当前时间字符串，格式：YYYYMMDD_HHMMSS"""
    return time.strftime("%Y%m%d_%H%M%S")

def get_click_coordinate_by_bbox(bbox: List[float], screen_width: int = 1920, screen_height: int = 1080) -> Tuple[int, int]:
    """
    从bbox获取点击坐标，返回bbox中心点的像素坐标
    
    Args:
        bbox: bounding box坐标，格式为[x1, y1, x2, y2]，值为0-1之间的比例
        screen_width: 屏幕宽度，默认1920
        screen_height: 屏幕高度，默认1080
    
    Returns:
        tuple: (x, y) 像素坐标，表示bbox的中心点
    
    About bbox:
    bbox是一个边界框(Bounding Box)，用来标识图像中某个UI元素的位置和大小。
    在OmniParser项目中，bbox有两种主要类型：
    1. OCR文本框 - 标识检测到的文字区域
    2. 图标框 - 标识检测到的可交互UI元素（按钮、图标等）
    
    bbox格式说明：
    - [x1, y1, x2, y2] - xyxy格式，表示左上角和右下角坐标
    - 坐标值是相对于图像尺寸的比例值（0-1之间）
    - x1, y1: 左上角坐标比例
    - x2, y2: 右下角坐标比例
    
    与手机/屏幕坐标的关系：
    - bbox坐标是图像相对坐标（比例）
    - 需要乘以实际屏幕尺寸得到绝对像素坐标
    - 用于UI自动化中的点击操作
    """
    if len(bbox) != 4:
        raise ValueError("bbox必须包含4个元素: [x1, y1, x2, y2]")
    
    x1, y1, x2, y2 = bbox
    
    # 验证bbox值是否在合理范围内
    if not all(0 <= coord <= 1 for coord in bbox):
        raise ValueError("bbox坐标必须在0-1之间")
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox坐标不合法: x1应小于x2，y1应小于y2")
    
    # 计算bbox中心点的比例坐标
    center_x_ratio = (x1 + x2) / 2
    center_y_ratio = (y1 + y2) / 2
    
    # 转换为实际像素坐标
    pixel_x = int(center_x_ratio * screen_width)
    pixel_y = int(center_y_ratio * screen_height)
    
    return pixel_x, pixel_y

def take_screenshot_by_coordinate(
    screenshot_name: str = None, 
    coordinate: Union[List[float], List[int], None] = None,
    screen_width: int = 1920,
    screen_height: int = 1080,
    padding: int = 20
) -> str:
    """
    根据坐标截取屏幕区域
    
    Args:
        screenshot_name: 截图文件名，如果为None则自动生成
        coordinate: 坐标信息，支持多种格式：
                   - bbox格式: [x1, y1, x2, y2] 比例坐标(0-1)
                   - 像素坐标: [x1, y1, x2, y2] 绝对坐标
                   - None: 截取全屏
        screen_width: 屏幕宽度
        screen_height: 屏幕高度
        padding: 截图区域的边距扩展（像素）
    
    Returns:
        str: 保存的截图文件路径
    
    示例用法:
    # 截取全屏
    take_screenshot_by_coordinate()
    
    # 根据bbox截取特定区域（比如"Home 1"按钮区域）
    bbox = [0.1, 0.2, 0.3, 0.4]  # 假设这是Home按钮的bbox
    take_screenshot_by_coordinate(coordinate=bbox)
    
    # 根据像素坐标截取
    pixel_coords = [100, 100, 300, 200]
    take_screenshot_by_coordinate(coordinate=pixel_coords)
    """
    
    # 生成文件名
    if screenshot_name is None:
        random_str = get_random_string()
        current_time = get_current_time()
        screenshot_name = f"screenshot_{random_str}_{current_time}.png"
    
    # 确保文件名以.png结尾
    if not screenshot_name.endswith('.png'):
        screenshot_name += '.png'
    
    try:
        if coordinate is None:
            # 截取全屏
            screenshot = ImageGrab.grab()
            print(f"截取全屏: {screen_width}x{screen_height}")
        else:
            # 处理坐标
            if len(coordinate) != 4:
                raise ValueError("坐标必须包含4个元素: [x1, y1, x2, y2]")
            
            x1, y1, x2, y2 = coordinate
            
            # 判断是比例坐标还是像素坐标
            if all(0 <= coord <= 1 for coord in coordinate):
                # 比例坐标，转换为像素坐标
                x1 = int(x1 * screen_width)
                y1 = int(y1 * screen_height)
                x2 = int(x2 * screen_width)
                y2 = int(y2 * screen_height)
                print(f"检测到bbox比例坐标，转换为像素坐标: ({x1}, {y1}, {x2}, {y2})")
            else:
                # 已经是像素坐标
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(f"使用像素坐标: ({x1}, {y1}, {x2}, {y2})")
            
            # 添加边距并确保坐标在屏幕范围内
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(screen_width, x2 + padding)
            y2 = min(screen_height, y2 + padding)
            
            # 截取指定区域
            screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            print(f"截取区域: ({x1}, {y1}, {x2}, {y2}), 尺寸: {x2-x1}x{y2-y1}")
        
        # 保存截图
        screenshot.save(screenshot_name)
        print(f"截图已保存: {screenshot_name}")
        
        return screenshot_name
        
    except Exception as e:
        print(f"截图失败: {str(e)}")
        raise

def bbox_to_pixel_coordinates(bbox: List[float], image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """
    将bbox比例坐标转换为像素坐标
    
    Args:
        bbox: [x1, y1, x2, y2] 比例坐标
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        tuple: (x1, y1, x2, y2) 像素坐标
    """
    x1, y1, x2, y2 = bbox
    pixel_x1 = int(x1 * image_width)
    pixel_y1 = int(y1 * image_height)
    pixel_x2 = int(x2 * image_width)
    pixel_y2 = int(y2 * image_height)
    
    return pixel_x1, pixel_y1, pixel_x2, pixel_y2

def get_bbox_area(bbox: List[float], image_width: int, image_height: int) -> int:
    """
    计算bbox的像素面积
    
    Args:
        bbox: [x1, y1, x2, y2] 比例坐标
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        int: bbox的像素面积
    """
    x1, y1, x2, y2 = bbox_to_pixel_coordinates(bbox, image_width, image_height)
    return (x2 - x1) * (y2 - y1)

def find_element_by_content(parsed_content_list: List[Dict], content_keyword: str) -> Dict:
    """
    根据内容关键词查找UI元素
    
    Args:
        parsed_content_list: OmniParser解析结果列表
        content_keyword: 要搜索的内容关键词
    
    Returns:
        dict: 匹配的元素信息，包含bbox等
    """
    for element in parsed_content_list:
        if element.get('content') and content_keyword.lower() in element['content'].lower():
            return element
    return None

# 使用示例和测试函数
def demo_usage():
    """
    演示函数使用方法
    """
    print("=== OmniParser bbox坐标系统说明 ===")
    print("1. bbox是什么？")
    print("   - bbox (bounding box) 是边界框，用来标识UI元素的位置")
    print("   - 格式: [x1, y1, x2, y2] - 左上角和右下角坐标")
    print("   - 坐标值是0-1之间的比例，相对于图像尺寸")
    print()
    
    print("2. 与手机/屏幕坐标的关系：")
    print("   - bbox使用相对坐标（比例），与设备无关")
    print("   - 实际点击需要转换为绝对像素坐标")
    print("   - 公式: 像素坐标 = bbox比例 × 屏幕尺寸")
    print()
    
    print("3. 使用示例：")
    
    # 示例1：模拟Home按钮的bbox
    home_bbox = [0.1, 0.85, 0.25, 0.95]  # 假设Home按钮在屏幕左下角
    print(f"   Home按钮bbox: {home_bbox}")
    
    # 获取点击坐标
    click_x, click_y = get_click_coordinate_by_bbox(home_bbox, 1920, 1080)
    print(f"   点击坐标: ({click_x}, {click_y})")
    
    # 截取Home按钮区域
    screenshot_file = take_screenshot_by_coordinate(
        screenshot_name="home_button_area",
        coordinate=home_bbox,
        screen_width=1920,
        screen_height=1080,
        padding=50  # 增加50像素边距
    )
    print(f"   截图保存为: {screenshot_file}")
    
    print()
    print("4. 实际使用场景：")
    print("   - 图像上的'Home 1'标注表示第1个Home相关的UI元素")
    print("   - 通过bbox可以精确定位和截取这个区域")
    print("   - 可用于UI自动化测试和交互")

if __name__ == "__main__":
    demo_usage() 
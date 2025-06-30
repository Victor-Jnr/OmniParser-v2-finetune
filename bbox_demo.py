"""
OmniParser bbox使用示例
演示如何理解和使用bbox坐标系统
"""

from screenshot_utils import get_click_coordinate_by_bbox, take_screenshot_by_coordinate, find_element_by_content
import json

def main():
    print("=== OmniParser项目答疑 ===\n")
    
    # 1. 解释bbox是什么
    print("1. bbox（边界框）详解：")
    print("   bbox = Bounding Box，是一个矩形框，用来标识UI元素的位置和大小")
    print("   格式：[x1, y1, x2, y2]")
    print("   - x1, y1: 左上角坐标（比例值 0-1）")
    print("   - x2, y2: 右下角坐标（比例值 0-1）")
    print("   - 比例值意味着与具体设备尺寸无关，具有通用性\n")
    
    # 2. 与手机坐标点的关系
    print("2. bbox与手机/屏幕坐标的关系：")
    print("   bbox使用归一化坐标（0-1之间的比例）")
    print("   转换公式：")
    print("   - 像素X = bbox_x × 屏幕宽度")
    print("   - 像素Y = bbox_y × 屏幕高度")
    print("   这样设计的好处：")
    print("   - 同一个UI元素在不同分辨率设备上的bbox值相同")
    print("   - 便于跨设备的UI自动化\n")
    
    # 3. 图片标注说明（以Home 1为例）
    print("3. 图片标注说明（如'Home 1'）：")
    print("   - 数字1表示这是第1个检测到的Home相关元素")
    print("   - 框框表示该元素的bbox边界")
    print("   - 在OmniParser中，每个检测到的UI元素都会分配一个ID\n")
    
    # 4. 实际示例演示
    print("4. 实际使用示例：")
    
    # 模拟一个Home按钮的bbox（假设位于屏幕底部中央）
    home_bbox = [0.4, 0.85, 0.6, 0.95]  # 屏幕底部中央区域
    print(f"   假设Home按钮的bbox: {home_bbox}")
    
    # 获取点击坐标
    click_x, click_y = get_click_coordinate_by_bbox(home_bbox, 1920, 1080)
    print(f"   转换为1920x1080屏幕的点击坐标: ({click_x}, {click_y})")
    
    # 不同分辨率下的坐标
    click_x_mobile, click_y_mobile = get_click_coordinate_by_bbox(home_bbox, 375, 812)
    print(f"   转换为375x812手机屏幕的点击坐标: ({click_x_mobile}, {click_y_mobile})")
    
    print("\n   可以看到，同一个bbox在不同分辨率下会得到不同的像素坐标")
    print("   但相对位置保持一致（都在屏幕底部中央）\n")
    
    # 5. 截图功能演示
    print("5. 如何在框框范围内截图：")
    print("   使用 take_screenshot_by_coordinate 函数")
    print("   示例代码：")
    print(f"   screenshot_file = take_screenshot_by_coordinate(")
    print(f"       screenshot_name='home_button_area',")
    print(f"       coordinate={home_bbox},")
    print(f"       screen_width=1920,")
    print(f"       screen_height=1080,")
    print(f"       padding=20  # 在bbox周围增加20像素边距")
    print(f"   )")
    print("\n   这会截取Home按钮区域及其周围20像素的范围\n")
    
    # 6. 实际项目中的数据示例
    print("6. 项目中的实际数据结构：")
    example_element = {
        "type": "icon",
        "bbox": [0.4, 0.85, 0.6, 0.95],
        "interactivity": True,
        "content": "Home",
        "source": "box_yolo_content_ocr"
    }
    print("   元素数据结构：")
    for key, value in example_element.items():
        print(f"   - {key}: {value}")
    print("\n   说明：")
    print("   - type: 元素类型（icon/text）")
    print("   - bbox: 边界框坐标")
    print("   - interactivity: 是否可交互")
    print("   - content: 元素内容/描述")
    print("   - source: 数据来源\n")
    
    # 7. 完整工作流程
    print("7. 完整使用流程：")
    print("   步骤1: 使用OmniParser解析屏幕截图")
    print("   步骤2: 从解析结果中找到目标元素（如Home按钮）")
    print("   步骤3: 提取元素的bbox坐标")
    print("   步骤4: 使用 get_click_coordinate_by_bbox 获取点击坐标")
    print("   步骤5: 使用 take_screenshot_by_coordinate 截取特定区域")
    print("   步骤6: 执行自动化操作（点击、验证等）")
    
def demo_with_real_data():
    """使用真实数据演示"""
    print("\n=== 使用真实数据演示 ===")
    
    # 模拟从OmniParser获取的解析结果
    parsed_content_list = [
        {
            "type": "text",
            "bbox": [0.05, 0.10, 0.86, 0.13],
            "interactivity": False,
            "content": "CloudStream-WiFi network details",
            "source": "box_ocr_content_ocr"
        },
        {
            "type": "icon", 
            "bbox": [0.82, 0.001, 0.97, 0.04],
            "interactivity": True,
            "content": "Close button",
            "source": "box_yolo_content_yolo"
        },
        {
            "type": "icon",
            "bbox": [0.40, 0.85, 0.60, 0.95],
            "interactivity": True, 
            "content": "Home",
            "source": "box_yolo_content_yolo"
        }
    ]
    
    print("解析结果中包含的元素：")
    for i, element in enumerate(parsed_content_list):
        print(f"   元素{i+1}: {element['content']} - bbox: {element['bbox']}")
    
    # 查找Home按钮
    home_element = find_element_by_content(parsed_content_list, "Home")
    if home_element:
        print(f"\n找到Home元素: {home_element['content']}")
        home_bbox = home_element['bbox']
        
        # 获取点击坐标
        click_x, click_y = get_click_coordinate_by_bbox(home_bbox, 1920, 1080)
        print(f"点击坐标: ({click_x}, {click_y})")
        
        # 演示如何截图（实际运行时会创建截图文件）
        print(f"\n如果要截取Home按钮区域，可以这样调用：")
        print(f"take_screenshot_by_coordinate('home_area.png', {home_bbox})")
    
    print("\n=== 总结 ===")
    print("通过这个项目，你可以：")
    print("1. 自动识别屏幕上的UI元素")
    print("2. 获取每个元素的精确位置（bbox）")
    print("3. 将bbox转换为可点击的像素坐标")
    print("4. 截取特定区域的屏幕图像")
    print("5. 实现跨设备的UI自动化")

if __name__ == "__main__":
    main()
    demo_with_real_data() 
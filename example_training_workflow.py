#!/usr/bin/env python3
"""
示例训练工作流程
演示如何使用OmniParser输出来训练自定义模型
"""

import os
import json
from pathlib import Path
from demo import main as demo_main
from collect_training_data import TrainingDataCollector
from finetune_omniparser_models import prepare_training_data_from_omniparser_output, YOLOTrainer, Florence2Trainer

def example_data_collection():
    """示例：从图像收集训练数据"""
    print("=== 步骤1：收集训练数据 ===")
    
    # 创建示例目录结构
    raw_images_dir = "example_raw_images"
    training_data_dir = "example_training_data"
    
    # 检查是否有示例图像
    if not os.path.exists(raw_images_dir):
        print(f"请创建目录 {raw_images_dir} 并放入您的训练图像")
        return None, None
    
    # 使用现有图像作为示例
    sample_images = ["imgs/teams.png", "imgs/word.png", "imgs/windows_home.png"]
    os.makedirs(raw_images_dir, exist_ok=True)
    
    for img in sample_images:
        if os.path.exists(img):
            import shutil
            shutil.copy2(img, raw_images_dir)
    
    # 收集训练数据
    collector = TrainingDataCollector()
    results = collector.process_image_directory(
        raw_images_dir,
        training_data_dir,
        manual_correction=False,  # 为简化示例，不使用手动校正
        box_threshold=0.05
    )
    
    return results, training_data_dir

def example_data_preparation(results, training_data_dir):
    """示例：准备训练数据"""
    print("\n=== 步骤2：准备训练数据格式 ===")
    
    if not results:
        print("没有可用的处理结果")
        return None, None
    
    # 准备训练数据
    yolo_config, florence_data = prepare_training_data_from_omniparser_output(
        training_data_dir, results
    )
    
    print(f"YOLO配置文件: {yolo_config}")
    print(f"Florence2数据样본数: {len(florence_data)}")
    
    return yolo_config, florence_data

def example_model_training(yolo_config, florence_data, training_data_dir):
    """示例：训练模型"""
    print("\n=== 步骤3：训练模型 ===")
    
    # 训练YOLO模型 (简化版本，少量epoch用于演示)
    if yolo_config and os.path.exists(yolo_config):
        print("开始训练YOLO模型...")
        yolo_trainer = YOLOTrainer()
        try:
            yolo_trainer.train(yolo_config, epochs=5)  # 演示用少量epoch
            print("YOLO模型训练完成!")
        except Exception as e:
            print(f"YOLO训练失败: {e}")
    
    # 训练Florence2模型 (如果有足够数据)
    if florence_data and len(florence_data) > 5:
        print("开始训练Florence2模型...")
        florence_trainer = Florence2Trainer()
        try:
            florence_trainer.train(florence_data, epochs=2, batch_size=2)  # 演示用参数
            print("Florence2模型训练完成!")
        except Exception as e:
            print(f"Florence2训练失败: {e}")
    else:
        print("Florence2数据不足，跳过训练")

def example_model_testing():
    """示例：测试训练好的模型"""
    print("\n=== 步骤4：测试训练后的模型 ===")
    
    test_image = "imgs/teams.png"
    if os.path.exists(test_image):
        print(f"测试图像: {test_image}")
        try:
            result = demo_main(test_image)
            print("模型测试完成!")
        except Exception as e:
            print(f"模型测试失败: {e}")
    else:
        print("没有找到测试图像")

def main():
    """主函数：演示完整的训练工作流程"""
    print("OmniParser模型训练工作流程演示")
    print("="*50)
    
    try:
        # 步骤1：收集数据
        results, training_data_dir = example_data_collection()
        
        if results:
            # 步骤2：准备训练数据
            yolo_config, florence_data = example_data_preparation(results, training_data_dir)
            
            # 步骤3：训练模型
            example_model_training(yolo_config, florence_data, training_data_dir)
            
            # 步骤4：测试模型
            example_model_testing()
            
            print("\n" + "="*50)
            print("工作流程演示完成!")
            print("\n实际使用建议:")
            print("1. 准备更多高质量的训练图像")
            print("2. 使用手动校正功能改善标注质量")
            print("3. 增加训练轮次获得更好的效果")
            print("4. 根据您的具体应用场景调整参数")
            
        else:
            print("数据收集失败，请检查图像目录和模型文件")
            
    except Exception as e:
        print(f"工作流程执行失败: {e}")
        print("请检查：")
        print("1. 是否安装了所有必要的依赖")
        print("2. 是否有可用的GPU（如果使用CUDA）")
        print("3. 是否有足够的磁盘空间")

if __name__ == "__main__":
    main() 
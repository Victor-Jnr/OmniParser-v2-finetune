#!/usr/bin/env python3
"""
模型层结构检查工具
用于分析 Florence2 模型的层结构，帮助确定微调策略
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import argparse

def analyze_model_structure(model_path="weights/icon_caption_florence"):
    """分析模型结构，显示各层参数信息"""
    
    print(f"Loading model from: {model_path}")
    
    try:
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )
        
        print(f"✓ Model loaded successfully")
        print(f"Model type: {model.config.model_type}")
        
    except Exception as e:
        print(f"Error loading local model: {e}")
        print("Falling back to base model...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    
    # 分析模型结构
    print("\n" + "="*80)
    print("MODEL STRUCTURE ANALYSIS")
    print("="*80)
    
    total_params = 0
    layer_groups = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # 按层分组
        if 'language_model' in name:
            if 'lm_head' in name:
                group = 'language_model.lm_head'
            elif 'encoder.layers.' in name:
                # 提取层号
                layer_num = name.split('encoder.layers.')[1].split('.')[0]
                group = f'language_model.encoder.layers.{layer_num}'
            elif 'embed' in name:
                group = 'language_model.embeddings'
            else:
                group = 'language_model.other'
        elif 'vision_tower' in name:
            group = 'vision_tower'
        elif 'projector' in name:
            group = 'projector'
        else:
            group = 'other'
        
        if group not in layer_groups:
            layer_groups[group] = {'params': 0, 'layers': []}
        
        layer_groups[group]['params'] += param_count
        layer_groups[group]['layers'].append((name, param_count))
    
    # 显示汇总信息
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Size: {total_params * 4 / (1024**3):.2f} GB (float32)")
    
    # 按组显示参数
    print(f"\n{'Group':<30} {'Parameters':<15} {'Percentage':<12} {'Size (MB)':<10}")
    print("-" * 70)
    
    for group, info in sorted(layer_groups.items()):
        params = info['params']
        percentage = (params / total_params) * 100
        size_mb = params * 4 / (1024**2)  # float32 = 4 bytes
        print(f"{group:<30} {params:<15,} {percentage:<12.2f}% {size_mb:<10.1f}")
    
    # 显示当前微调策略
    print(f"\n{'='*80}")
    print("CURRENT FINE-TUNING STRATEGY ANALYSIS")
    print("="*80)
    
    trainable_keywords = [
        'language_model.lm_head',
        'language_model.model.encoder.layers.5',
        'language_model.model.encoder.layers.4',
        'projector'
    ]
    
    trainable_params = 0
    frozen_params = 0
    
    print(f"\n{'Layer':<50} {'Status':<12} {'Parameters':<15}")
    print("-" * 80)
    
    for group, info in sorted(layer_groups.items()):
        is_trainable = any(keyword in group for keyword in trainable_keywords)
        
        # 额外检查具体的层名称
        if not is_trainable:
            for layer_name, _ in info['layers']:
                if any(keyword in layer_name for keyword in trainable_keywords):
                    is_trainable = True
                    break
        
        status = "TRAINABLE" if is_trainable else "FROZEN"
        params = info['params']
        
        if is_trainable:
            trainable_params += params
        else:
            frozen_params += params
        
        print(f"{group:<50} {status:<12} {params:<15,}")
        
        # 显示匹配的具体层
        if is_trainable:
            matching_layers = []
            for layer_name, layer_params in info['layers']:
                if any(keyword in layer_name for keyword in trainable_keywords):
                    matching_layers.append(f"  → {layer_name}")
            if matching_layers:
                for match in matching_layers[:3]:  # 只显示前3个
                    print(match)
    
    print("-" * 80)
    print(f"{'TOTAL TRAINABLE':<50} {'✓':<12} {trainable_params:<15,}")
    print(f"{'TOTAL FROZEN':<50} {'✗':<12} {frozen_params:<15,}")
    
    trainable_percentage = (trainable_params / total_params) * 100
    frozen_percentage = (frozen_params / total_params) * 100
    
    print(f"\nTrainable: {trainable_percentage:.2f}% ({trainable_params:,} parameters)")
    print(f"Frozen: {frozen_percentage:.2f}% ({frozen_params:,} parameters)")
    
    # 内存估算
    trainable_memory = trainable_params * 4 / (1024**2)  # MB
    total_memory = total_params * 4 / (1024**2)  # MB
    
    print(f"\nMemory for trainable parameters: {trainable_memory:.1f} MB")
    print(f"Memory savings vs full fine-tuning: {total_memory - trainable_memory:.1f} MB ({frozen_percentage:.1f}% reduction)")
    
    # 详细层信息（可选）
    return model, layer_groups

def show_detailed_layers(layer_groups, show_details=False):
    """显示详细的层信息"""
    if not show_details:
        return
        
    print(f"\n{'='*80}")
    print("DETAILED LAYER INFORMATION")
    print("="*80)
    
    for group, info in sorted(layer_groups.items()):
        print(f"\n{group} ({info['params']:,} parameters):")
        for name, params in info['layers'][:5]:  # 只显示前5个
            print(f"  {name:<60} {params:>10,}")
        if len(info['layers']) > 5:
            print(f"  ... and {len(info['layers']) - 5} more layers")

def compare_with_lora():
    """比较当前方法和LoRA的参数效率"""
    print(f"\n{'='*80}")
    print("LoRA COMPARISON")
    print("="*80)
    
    print("如果使用 LoRA (rank=16):")
    print("- 目标层: 注意力层的 q_proj, v_proj")
    print("- 每层 LoRA 参数: ~65K (取决于隐藏层大小)")
    print("- 总 LoRA 参数: <1M (约0.1%的模型参数)")
    print("- 内存需求: <10 MB")
    print()
    print("当前层冻结方法 vs LoRA:")
    print("- 当前方法: 更直接，易于理解和调试")
    print("- LoRA: 更节省内存，支持多任务适配器")
    print("- 建议: 资源充足时用当前方法，大规模部署时考虑LoRA")

def show_layer(model):

    # --- 2. 查找所有线性层作为 LoRA 的目标模块 ---
    # 我们使用一个 set 来存储模块名称，以自动处理重复项
    lora_target_modules = set()

    # 遍历模型的所有模块 (module) 及其名称 (name)
    for name, module in model.named_modules():
        # 检查当前模块是否是 torch.nn.Linear 类的实例
        # 这是 LoRA 最主要的应用对象
        if isinstance(module, torch.nn.Linear):
            # 如果是线性层，就将其名称添加到 set 中
            lora_target_modules.add(name)

    # --- 3. 打印结果 ---
    print(f"在模型 {model} 中找到 {len(lora_target_modules)} 个潜在的 LoRA 目标模块:")
    # 为了方便查看，我们将 set 转换为 list 并排序后打印
    for module_name in sorted(list(lora_target_modules)):
        print(module_name)

    # --- 4. 建议的实践 ---
    # 在实践中，通常会选择注意力机制中的 q_proj 和 v_proj
    # 我们可以从上面的完整列表中筛选出这些模块
    recommended_modules = [name for name in lora_target_modules if "q_proj" in name or "v_proj" in name or "qkv" in name]

    print("\n--- 建议的目标模块 (通常选择注意力层中的 q_proj 和 v_proj) ---")
    for module_name in sorted(recommended_modules):
        print(module_name)

def main():
    parser = argparse.ArgumentParser(description="分析 Florence2 模型层结构")
    parser.add_argument("--model_path", default="weights/icon_caption_florence",
                       help="模型路径")
    parser.add_argument("--detailed", action="store_true",
                       help="显示详细层信息") # paramter info
    parser.add_argument("--compare_lora", action="store_true",
                       help="显示与LoRA的对比")
    parser.add_argument("--show_layer", action="store_true",
                       help="显示所有层信息") #generate by gemini 
    args = parser.parse_args()
    
    # 分析模型结构
    model, layer_groups = analyze_model_structure(args.model_path)
    
    # 显示详细信息
    if args.detailed:
        show_detailed_layers(layer_groups, show_details=True)
    if args.show_layer:
        show_layer(model)
    # LoRA对比
    if args.compare_lora:
        compare_with_lora()
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("="*80)
    print("1. 当前策略适合快速实验和概念验证")
    print("2. 如需更高效率，考虑实现 LoRA 方法")
    print("3. 层选择基于任务相关性：projector(跨模态) > lm_head(输出) > top_layers(语义)")
    print("4. 详细分析请参阅: FINETUNING_METHODS_README.md")

if __name__ == "__main__":
    main()
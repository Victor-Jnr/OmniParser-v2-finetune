# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>
<!-- <a href="https://trendshift.io/repositories/12975" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12975" alt="microsoft%2FOmniParser | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a> -->

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[Models V2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[Models V1.5](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace Space Demo](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface.

## News

- [2025/3] We support local logging of trajecotry so that you can use OmniParser+OmniTool to build training data pipeline for your favorate agent in your domain. [Documentation WIP]
- [2025/3] We are gradually adding multi agents orchstration and improving user interface in OmniTool for better experience.
- [2025/2] We release OmniParser V2 [checkpoints](https://huggingface.co/microsoft/OmniParser-v2.0). [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] We introduce OmniTool: Control a Windows 11 VM with OmniParser + your vision model of choice. OmniTool supports out of the box the following large language models - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use. [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 is coming. We achieve new state of the art results 39.5% on the new grounding benchmark [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) with OmniParser v2 (will be released soon)! Read more details [here](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md).
- [2024/11] We release an updated version, OmniParser V1.5 which features 1) more fine grained/small icon detection, 2) prediction of whether each screen element is interactable or not. Examples in the demo.ipynb.
- [2024/10] OmniParser was the #1 trending model on huggingface model hub (starting 10/29/2024).
- [2024/10] Feel free to checkout our demo on [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)! (stay tuned for OmniParser + Claude Computer Use)
- [2024/10] Both Interactive Region Detection Model and Icon functional description model are released! [Hugginface models](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser achieves the best performance on [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)!

## Install

First clone the repo, and then install environment:

```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Ensure you have the V2 weights downloaded in weights folder (ensure caption weights folder is called icon_caption_florence). If not download them with:

```
   # download the model checkpoints to local directory OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [deprecated]
Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2. 

For v1: 
convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py

For v1.5: 
download 'model_v1_5.pt' from https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, make a new dir: weights/icon_detect_v1_5, and put it inside the folder. No weight conversion is needed. 
``` -->

## Examples:

We put together a few simple examples in the demo.ipynb.

## Gradio Demo

To run gradio demo, simply run:

```python
python gradio_demo.py
```

## Model Weights License

For the model checkpoints on huggingface model hub, please note that icon_detect model is under AGPL license since it is a license inherited from the original yolo model. And icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model: https://huggingface.co/microsoft/OmniParser.

## 📚 Citation

Our technical report can be found [here](https://arxiv.org/abs/2408.00203).
If you find our work useful, please consider citing our work:

```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```

## 技术架构详解 - 三层提取机制

OmniParser采用了一个三层级联的检测和识别架构，分别处理不同类型的界面元素：

### 1. 完整的处理流程

```
输入图片 → PaddleOCR文本检测 → YOLO图标检测 → 重叠处理与融合 → Florence2语义理解 → 输出结构化结果
```

具体流程如下：

1. **第一层：OCR文本检测 (PaddleOCR)**

   - 使用PaddleOCR识别图像中的所有文本区域
   - 提取文本内容和边界框坐标
   - 生成 `box_ocr_content_ocr` 类型的元素
2. **第二层：图标区域检测 (YOLO)**

   - 使用训练好的YOLO模型检测可交互的图标区域
   - 输出边界框坐标，但不包含语义内容
   - 此时图标元素的content为None
3. **第三层：重叠处理与语义融合**

   - 使用`remove_overlap_new()`函数处理OCR和YOLO检测结果的重叠
   - 如果OCR文本在YOLO检测的图标内部，则合并为 `box_yolo_content_ocr`
   - 如果YOLO检测到的图标没有OCR文本，保留为 `box_yolo_content_yolo`
4. **第四层：语义内容生成 (Florence2)**

   - 对于content为None的图标区域，裁剪出64x64的图像片段
   - 使用Florence2模型生成语义描述
   - 更新图标的content字段

### 2. 三种Source类型的含义

- **`box_ocr_content_ocr`**: 纯OCR识别的文本区域，interactivity=False
- **`box_yolo_content_ocr`**: YOLO检测到的图标区域内包含OCR文本，合并了两者信息，interactivity=True
- **`box_yolo_content_yolo`**: YOLO检测到的图标区域，由Florence2生成语义描述，interactivity=True

### 3. YOLO如何输出Content

YOLO本身只检测边界框，不直接输出content。Content的生成过程是：

1. **重叠检测阶段**：如果YOLO框内有OCR文本，直接使用OCR的文本作为content
2. **语义生成阶段**：对于没有OCR文本的YOLO框，通过以下步骤生成content：
   ```python
   # 裁剪图标区域
   cropped_image = image_source[ymin:ymax, xmin:xmax, :]
   cropped_image = cv2.resize(cropped_image, (64, 64))

   # 使用Florence2生成描述
   inputs = processor(images=batch, text=["<CAPTION>"], return_tensors="pt")
   generated_ids = model.generate(input_ids=inputs["input_ids"], 
                                 pixel_values=inputs["pixel_values"], 
                                 max_new_tokens=20, num_beams=1)
   ```

### 4. 提高APK探测准确度的建议

**问题分析**：单纯训练YOLO模型有一定局限性

**推荐方案**：

1. **数据增强**：

   - 使用项目现有输出作为基础数据
   - 人工修正边界框和标签
   - 增加移动端界面的训练样本
2. **模型组合优化**：

   - 保持三层架构，重点优化YOLO检测准确率
   - 针对APK界面特点，调整检测阈值 (BOX_THRESHOLD, IoU阈值)
   - 考虑使用更大的图像输入尺寸 (imgsz参数)
3. **训练策略**：

   ```python
   # 数据准备流程
   原始APK截图 → OmniParser处理 → 人工校正 → 转换为YOLO训练格式 → 增量训练
   ```
4. **特殊优化**：

   - 增强Florence2模型对移动界面元素的描述能力
   - 优化OCR参数以更好识别移动端文本
   - 调整重叠处理的IoU阈值，适应移动端界面特点

**关键代码位置**：

- YOLO训练：`weights/icon_detect/`
- 重叠处理：`util/utils.py:remove_overlap_new()`
- 语义生成：`util/utils.py:get_parsed_content_icon()`

### 5. 核心代码示例

**重叠处理逻辑**：

```python
def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    # OCR文本在YOLO图标内部的判断
    if is_inside(box3, box1): # ocr inside icon
        ocr_labels += box3_elem['content'] + ' '
        filtered_boxes.remove(box3_elem)
    elif is_inside(box1, box3): # icon inside ocr
        box_added = True  # 不添加此图标框
        break
  
    # 根据是否有OCR文本设置不同的source
    if ocr_labels:
        filtered_boxes.append({
            'type': 'icon', 
            'bbox': box1_elem['bbox'], 
            'interactivity': True, 
            'content': ocr_labels, 
            'source': 'box_yolo_content_ocr'
        })
    else:
        filtered_boxes.append({
            'type': 'icon', 
            'bbox': box1_elem['bbox'], 
            'interactivity': True, 
            'content': None, 
            'source': 'box_yolo_content_yolo'
        })
```

**完整处理流程**：

```python
# 主处理函数 get_som_labeled_img() 的核心步骤
def get_som_labeled_img(image_source, model, ...):
    # 1. YOLO检测图标
    xyxy, logits, phrases = predict_yolo(model, image_source, ...)
  
    # 2. 创建OCR和YOLO元素
    ocr_bbox_elem = [{'type': 'text', 'bbox': box, 'content': txt, 'source': 'box_ocr_content_ocr'} 
                     for box, txt in zip(ocr_bbox, ocr_text)]
    xyxy_elem = [{'type': 'icon', 'bbox': box, 'content': None} 
                 for box in xyxy.tolist()]
  
    # 3. 处理重叠并融合
    filtered_boxes = remove_overlap_new(xyxy_elem, iou_threshold, ocr_bbox_elem)
  
    # 4. 对content=None的图标生成语义描述
    if use_local_semantics:
        parsed_content_icon = get_parsed_content_icon(filtered_boxes, ...)
        # 填充空content
        for box in filtered_boxes_elem:
            if box['content'] is None:
                box['content'] = parsed_content_icon.pop(0)
```

### 6. 流程可视化

上述流程可以用以下图表表示：

```
输入图片
├── PaddleOCR文本检测 → OCR边界框 + 文本内容
└── YOLO图标检测 → YOLO边界框 (content=None)
                    ↓
              remove_overlap_new() 重叠处理
                    ↓
         ┌─────────────────────────┐
         ├── OCR文本在YOLO框内? ──┤
         └─────────────────────────┘
         ↓是                    ↓否
box_yolo_content_ocr     box_yolo_content_yolo
(合并OCR文本)              (content=None)
                              ↓
                      Florence2语义生成
                              ↓
                       更新content字段
                              ↓
                      最终结构化输出
```

### 7. Florence2模型的详细工作原理

Florence2模型在OmniParser中扮演着关键的语义理解角色，专门负责为没有OCR文本的图标区域生成语义描述。

#### 7.1 Florence2的输入处理

```python
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, ...):
    # 1. 提取需要处理的图标区域（content=None的YOLO检测框）
    non_ocr_boxes = filtered_boxes[starting_idx:]  # 跳过已有content的OCR区域
  
    # 2. 裁剪并预处理图像
    croped_pil_image = []
    for coord in non_ocr_boxes:
        # 将相对坐标转换为像素坐标
        xmin, xmax = int(coord[0]*width), int(coord[2]*width)
        ymin, ymax = int(coord[1]*height), int(coord[3]*height)
      
        # 裁剪出图标区域并调整为64x64标准尺寸
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        cropped_image = cv2.resize(cropped_image, (64, 64))
        croped_pil_image.append(to_pil(cropped_image))
```

#### 7.2 Florence2的推理过程

```python
    # 3. 批量处理图像（优化性能）
    model, processor = caption_model_processor['model'], caption_model_processor['processor']
  
    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i:i+batch_size]
      
        # 4. 准备输入（图像+提示词）
        inputs = processor(
            images=batch, 
            text=["<CAPTION>"] * len(batch),  # Florence2特有的提示格式
            return_tensors="pt", 
            do_resize=False
        ).to(device=device, dtype=torch.float16)
      
        # 5. 生成语义描述
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=20,  # 限制输出长度
            num_beams=1,        # 贪婪搜索
            do_sample=False     # 确定性输出
        )
      
        # 6. 解码生成的文本
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts.extend([gen.strip() for gen in generated_text])
```

#### 7.3 Florence2在不同场景中的应用

**输入示例**：64x64的图标裁剪图像

- 📁 文件夹图标 → "folder"
- ⚙️ 设置齿轮 → "settings gear icon"
- 🔍 搜索放大镜 → "magnifying glass search"
- ➕ 添加按钮 → "plus add button"

**实际输出示例**（来自项目数据）：

```json
{
  "type": "icon",
  "bbox": [0.8391380906105042, 0.16333389282226562, 0.9413251876831055, 0.19683626294136047],
  "interactivity": true,
  "content": "a loading or buffering indicator.",
  "source": "box_yolo_content_yolo"
}
```

#### 7.4 Florence2模型的优势

1. **专门优化**：Florence2是微软专门为视觉理解任务优化的多模态模型
2. **高效推理**：支持批量处理，64x64小图像处理速度快
3. **语义丰富**：能够识别图标的功能含义，而不仅仅是视觉特征
4. **上下文理解**：结合UI界面的上下文生成更准确的描述

#### 7.5 Florence2 vs OCR的协作关系


| 处理对象     | 处理方式          | 输出特点              | Source标记            |
| ------------ | ----------------- | --------------------- | --------------------- |
| 文本区域     | PaddleOCR直接识别 | 精确的文字内容        | box_ocr_content_ocr   |
| 有文字的图标 | OCR+YOLO融合      | "文字内容 + 图标属性" | box_yolo_content_ocr  |
| 纯图标区域   | Florence2语义理解 | 功能性描述            | box_yolo_content_yolo |

**关键差异**：

- **OCR**：识别"what is written"（写了什么）
- **Florence2**：理解"what does it do"（做什么用的）

----
New version

# OmniParser Chat Session Summary

## 🎯 Main Question

User asked about the **three-layer extraction mechanism** in OmniParser project, specifically:

- What is the actual extraction flow?
- How does YOLO output content?
- Training strategies for better APK detection accuracy

## 🔍 User's Initial Understanding (Corrected)

**Thought flow was:** Input image → extract positions → PaddleOCR → YOLO → final recognition → concatenate results

## ✅ Technical Analysis & Corrections

### OmniParser's Actual Architecture

**Real flow:** Input image → [PaddleOCR + YOLO parallel] → overlap processing/fusion → Florence2 semantic understanding → structured output

### Three-Layer Extraction Mechanism

1. **Layer 1:** PaddleOCR text detection (generates `box_ocr_content_ocr`)
2. **Layer 2:** YOLO icon detection (detects bounding boxes)
3. **Layer 3:** Overlap processing + Florence2 semantic understanding

### Three Source Types in Output

- `box_ocr_content_ocr`: Pure OCR text regions (interactivity=False)
- `box_yolo_content_ocr`: YOLO-detected icons containing OCR text (interactivity=True)
- `box_yolo_content_yolo`: YOLO-detected icons with Florence2-generated descriptions (interactivity=True)

### YOLO Content Generation Clarification

- **YOLO only detects bounding boxes** - doesn't directly output content
- Content comes from:
  - **Overlap detection:** Uses OCR text when YOLO box overlaps with OCR region
  - **Semantic generation:** Florence2 processes 64x64 cropped images for descriptions

## 🛠️ Training Solution Created

### Files Created

#### 1. `finetune_omniparser_models.py` - Main Training Script

- `YOLOTrainer` class for training icon detection
- `Florence2Trainer` class for training icon captioning
- `OmniParserDatasetConverter` for data format conversion
- Support for training both models separately or together

#### 2. `collect_training_data.py` - Data Collection Script

- `TrainingDataCollector` class using existing OmniParser to process images
- Manual correction interface via CSV export
- Batch processing capabilities

#### 3. `TRAINING_GUIDE.md` - Comprehensive Guide

- Data format specifications (YOLO format for detection, JSON for Florence2)
- Step-by-step training workflow
- APK-specific optimization strategies
- Troubleshooting guide

#### 4. `example_training_workflow.py` - Demonstration Script

Complete workflow example showing how to use all components

### Training Workflow

1. **Collect raw images** → process with existing OmniParser → generate training data
2. **Optional manual correction** via CSV editing
3. **Train models:** YOLO for detection + Florence2 for captioning
4. **Integration:** YOLO detects → Florence2 describes

## 🎯 APK Optimization Recommendations

- Use project output + manual correction for training data
- Maintain three-layer architecture rather than training only YOLO
- Adjust detection thresholds and IoU parameters for mobile interfaces
- Include diverse mobile interface samples in training data

## 🐛 Critical Issue: Florence模型训练问题

### 问题症状
经过finetune的Florence模型输出混乱：
- **Native输出**：准确的UI描述（`'M0,0L9,0 4.5,5z'`, `'P: 0:1'`, `'tool for cutting'`）
- **Finetuned输出**：不相关描述（`"person's face"`, `"camera in hand"`, `"No object detected"`）

### 根本原因
1. **灾难性遗忘**：学习率过高、缺乏层冻结策略
2. **模型目标退化**：从UI专业理解退化为通用图像描述
3. **训练策略问题**：所有层同时训练，破坏了预训练知识

### 解决方案
**创建了保守训练策略** (`finetune_omniparser_models_fixed.py`)：
- **层冻结**：冻结vision encoder，只训练顶层language model
- **更小学习率**：2e-6 (原来1e-5)
- **UI专用prompt**：修复`<CAPTION>`格式错误，添加UI上下文到答案
- **Early stopping**：防止过拟合
- **梯度裁剪**：稳定训练过程
- **🔧 使用本地模型**：基于`weights/icon_caption_florence_finetuned`而非在线模型

## 🎯 数据平衡功能优化

### 问题发现
用户发现训练数据分布不平衡问题：修改的数据很少，原始数据很多，导致模型被大量旧数据主导。

### 解决方案
**添加了数据平衡功能到 `collect_training_data.py`**：

#### 新功能特性
- **自动统计修改比例**：检查有多少数据被实际修改了
- **数据分布分析**：显示修改vs未修改数据的详细统计
- **智能数据平衡**：根据`old_percentage`参数随机删除多余的未修改数据
- **可配置比例**：用户可以指定保留多少百分比的原始数据

#### 使用方法
```bash
# 默认保留50%未修改数据
python collect_training_data.py --output_dir ./data --apply_corrections

# 只保留20%未修改数据（推荐用于大量原始数据场景）
python collect_training_data.py --output_dir ./data --apply_corrections --old_percentage 20

# 保留80%未修改数据（适合修改数据很多的场景）
python collect_training_data.py --output_dir ./data --apply_corrections --old_percentage 80
```

#### 输出示例
```
📊 Data Distribution Analysis:
  Modified elements: 45 (8.2%)
  Unchanged elements: 505 (91.8%)
  Total elements: 550

🎯 Data Balancing:
  Target unchanged elements: 110 (20%)
  Randomly removing 395 unchanged elements

✅ Final Data Distribution:
  Modified elements: 45 (29.0%)
  Unchanged elements: 110 (71.0%)
  Total elements: 155
```

## 📁 Key Architecture Files Analyzed

- `util/omniparser.py` - Main OmniParser class
- `util/utils.py` - Core processing functions
- `weights/icon_detect/` - YOLO model directory
- `weights/icon_caption_florence_finetuned/` - Florence2 model directory

## 🎉 Outcome

Created a complete training pipeline that:

- ✅ Correctly understands OmniParser's three-layer architecture
- ✅ Provides tools for collecting and preparing training data
- ✅ Supports training both YOLO detection and Florence2 captioning models
- ✅ Includes comprehensive documentation and examples
- ✅ Focuses on APK interface detection improvements
- ✅ **新增：数据平衡功能** - 智能调整训练数据分布

## 💡 Key Insights

- OmniParser uses **parallel processing** (PaddleOCR + YOLO), not sequential
- **YOLO only detects**, content generation is separate (overlap + Florence2)
- Training both models together maintains the **three-layer synergy**
- **Manual correction** capability is crucial for high-quality training data
- **数据平衡** 对防止模型被旧数据主导至关重要



# from ultralytics import YOLO
import os
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# utility function
import os
from openai import AzureOpenAI

import json
import sys
import os
import cv2
import numpy as np
import uuid
# %matplotlib inline
from matplotlib import pyplot as plt
import easyocr
from paddleocr import PaddleOCR
reader = easyocr.Reader(['en'])  # 初始化EasyOCR英文读取器
import multiprocessing as mp

# Optimize PaddleOCR for better CPU utilization
cpu_count = mp.cpu_count()  # 获取CPU核心数
paddle_ocr = PaddleOCR(
    lang='en',  # other lang also available
    use_angle_cls=False,  # 不使用文本方向分类器
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    show_log=False,  # 不显示日志
    max_batch_size=min(2048, cpu_count * 128),  # Scale batch size with CPU cores
    use_dilation=True,  # improves accuracy
    det_db_score_mode='slow',  # improves accuracy
    rec_batch_num=min(2048, cpu_count * 128),  # Scale batch size with CPU cores
    cpu_threads=cpu_count,  # Use all CPU cores
    enable_mkldnn=True,  # Enable Intel MKL-DNN for better CPU performance
    use_tensorrt=False,  # Disable TensorRT for CPU
    precision='fp32'  # Use FP32 for better compatibility
)
import time
import base64

import os
import ast
import torch
from typing import Tuple, List, Union
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from util.box_annotator import BoxAnnotator 

# from transformers import BitsAndBytesConfig

# quantization_config_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    """获取图像标题生成模型和处理器"""
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float32
        ) 
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float16
        ).to(device)
    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM 
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True)
        else:
            print(f'{model_name_or_path} model loaded to gpu')
            # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=quantization_config_4bit, torch_dtype=torch.float16, trust_remote_code=True).to(device)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    return {'model': model.to(device), 'processor': processor}


def get_yolo_model(model_path):
    """加载YOLO目标检测模型"""
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(model_path)
    return model


@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=256):
    """对图标区域进行内容识别和描述生成"""
    # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]  # 获取非OCR区域的框
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []  # 存储裁剪后的图像
    for i, coord in enumerate(non_ocr_boxes):
        try:
            # 将相对坐标转换为像素坐标
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]  # 裁剪图像
            cropped_image = cv2.resize(cropped_image, (64, 64))  # 调整大小为64x64
            croped_pil_image.append(to_pil(cropped_image))
        except:
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        # 根据模型类型设置提示词
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
    
    generated_texts = []  # 存储生成的文本描述
    device = model.device
    # 分批处理图像
    for i in range(0, len(croped_pil_image), batch_size):
        start = time.time()
        batch = croped_pil_image[i:i+batch_size]  # 当前批次的图像
        t1 = time.time()
        # 根据设备类型处理输入
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
        # 根据模型类型生成文本
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]  # 清理生成的文本
        generated_texts.extend(generated_text)
    
    return generated_texts



def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    """使用Phi3-Vision模型对图标区域进行内容识别"""
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]  # 跳过OCR区域的框
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []  # 存储裁剪后的图像
    for i, coord in enumerate(non_ocr_boxes):
        # 将相对坐标转换为像素坐标并裁剪图像
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}]  # 设置对话消息
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []  # 存储生成的文本描述

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    """移除重叠的边界框"""
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        """计算边界框面积"""
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        """计算两个边界框的交集面积"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        """计算两个边界框的IoU（交并比）"""
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        """判断box1是否在box2内部"""
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []  # 存储过滤后的边界框
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)  # 先添加OCR边界框
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            # keep the smaller box - 保留较小的框
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                # only add the box if it does not overlap with any ocr bbox - 只有不与OCR框重叠时才添加
                if not any(IoU(box1, box3) > iou_threshold and not is_inside(box1, box3) for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1): # ocr inside icon
                            # box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            # break
                        elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1_elem)
    return filtered_boxes # torch.tensor(filtered_boxes)


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases

def random_uuid():
    """生成8位随机UUID字符串"""
    uuid_str = uuid.uuid4().hex[:8]
    return str(uuid_str)

def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    if scale_img:
        # 使用指定图像尺寸进行预测
        result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        iou=iou_threshold, # default 0.7
        )
    else:
        # 使用原始图像尺寸进行预测
        result = model.predict(
        source=image,
        conf=box_threshold,
        iou=iou_threshold, # default 0.7
        )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space - 像素空间中的边界框
    conf = result[0].boxes.conf  # 置信度分数
    phrases = [str(i) for i in range(len(boxes))]  # 生成标签序号

    return boxes, conf, phrases

def int_box_area(box, w, h):
    """计算边界框在整数像素坐标下的面积"""
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]  # 转换为整数像素坐标
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])  # 计算面积
    return area

def get_som_labeled_img(image_source: Union[str, Image.Image], model=None, BOX_TRESHOLD=0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None, scale_img=False, imgsz=None, batch_size=None):
    """Process either an image path or Image object
    
    Args:
        image_source: Either a file path (str) or PIL Image object
        batch_size: Batch size for processing (auto-determined if None)
        ...
    """
    if isinstance(image_source, str):
        image_source = Image.open(image_source)  # 如果是路径则加载图像
    image_source = image_source.convert("RGB") # for CLIP - 转换为RGB格式
    w, h = image_source.size  # 获取图像尺寸
    if not imgsz:
        imgsz = (h, w)  # 设置默认图像尺寸
    
    # Auto-determine optimal batch size based on available resources
    if batch_size is None:
        if torch.cuda.is_available():
            # GPU: larger batch size, limited by memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            batch_size = min(512, int(gpu_memory_gb * 64))  # Scale with GPU memory - 根据GPU内存调整批次大小
        else:
            # CPU: moderate batch size, scale with CPU cores
            batch_size = min(256, cpu_count * 16)  # 根据CPU核心数调整批次大小
    
    print(f'Processing image size: {w}x{h}, batch_size: {batch_size}')
    # 使用YOLO模型进行目标检测
    xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)  # 归一化坐标到[0,1]范围
    image_source = np.asarray(image_source)  # 转换为numpy数组
    phrases = [str(i) for i in range(len(phrases))]  # 生成标签列表

    # annotate the image with labels - 为图像添加标签注释
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])  # 归一化OCR边界框坐标
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None

    # 创建OCR边界框元素字典
    if ocr_bbox and ocr_text:
        ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0]
    else:
        ocr_bbox_elem = [] 
    # 创建图标边界框元素字典
    xyxy_elem = [{'uuid':random_uuid(),'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
    # 移除重叠的边界框
    filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
    
    # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
    # 将过滤后的边界框按内容排序，有内容的在前，无内容的在后
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
    # get the index of the first 'content': None
    # 获取第一个无内容元素的索引
    starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
    filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])  # 提取边界框坐标
    print('len(filtered_boxes):', len(filtered_boxes), starting_idx)

    # get parsed icon local semantics - 获取图标的局部语义信息
    time1 = time.time()
    if use_local_semantics:
        caption_model = caption_model_processor['model']
        # 根据模型类型选择不同的处理方式
        if 'phi3_v' in caption_model.config.model_type: 
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=prompt,batch_size=batch_size)
        # 为OCR文本添加ID标识
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)  # 图标ID起始编号
        parsed_content_icon_ls = []
        # fill the filtered_boxes_elem None content with parsed_content_icon in order
        # 按顺序填充空内容的边界框元素
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None:
                box['content'] = parsed_content_icon.pop(0)
        # 为图标内容添加ID标识
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls  # 合并OCR和图标内容
    else:
        # 只使用OCR文本
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text
    print('time to get parsed content:', time.time()-time1)

    # 将边界框格式从xyxy转换为cxcywh（中心点+宽高）
    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]  # 生成标签序号列表
    
    # draw boxes - 绘制边界框
    if draw_bbox_config:
        # 使用自定义配置绘制注释
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        # 使用默认配置绘制注释
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
    
    # 将注释后的图像转换为PIL格式并编码为base64
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    
    # 如果需要相对坐标输出，将坐标归一化
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, filtered_boxes_elem


def get_xywh(input):
    """将边界框坐标转换为xywh格式（左上角坐标+宽高）"""
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    """将边界框坐标转换为xyxy格式（左上角和右下角坐标）"""
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh_yolo(input):
    """将YOLO格式的边界框坐标转换为xywh格式"""
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def check_ocr_box(image_source: Union[str, Image.Image], display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)  # 如果是路径则加载图像
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')  # 转换RGBA为RGB避免透明通道问题
    image_np = np.array(image_source)  # 转换为numpy数组
    w, h = image_source.size  # 获取图像尺寸
    
    # Optimize OCR processing
    if use_paddleocr:
        if easyocr_args is None:
            text_threshold = 0.5  # 默认文本阈值
        else:
            text_threshold = easyocr_args['text_threshold']
        
        # Use optimized PaddleOCR with better threading
        start_time = time.time()
        result = paddle_ocr.ocr(image_np, cls=False)[0]  # 进行OCR识别
        ocr_time = time.time() - start_time
        print(f'PaddleOCR processing time: {ocr_time:.2f}s')
        
        if result is None:
            coord, text = [], []  # 如果没有识别结果
        else:
            # 过滤置信度低于阈值的结果
            coord = [item[0] for item in result if item[1][1] > text_threshold]
            text = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        
        # Optimize EasyOCR args for better performance
        optimized_args = {
            'batch_size': min(32, cpu_count * 2),  # Scale with CPU cores
            'workers': cpu_count,  # Use all CPU cores
            **easyocr_args
        }
        
        start_time = time.time()
        result = reader.readtext(image_np, **optimized_args)  # 进行OCR识别
        ocr_time = time.time() - start_time
        print(f'EasyOCR processing time: {ocr_time:.2f}s')
        
        coord = [item[0] for item in result]  # 提取坐标
        text = [item[1] for item in result]   # 提取文本
    
    if display_img:
        # 显示图像和边界框
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)  # 绘制绿色矩形框
        #  matplotlib expects RGB
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        # 根据指定格式转换边界框坐标
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering
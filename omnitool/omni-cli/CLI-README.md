# Omni CLI 使用说明

Omni CLI 是 OmniParser 的命令行工具，支持三种输入方式来解析图像内容。

## 安装依赖

确保安装了必要的依赖：

```bash
pip install requests
```

或者安装完整的项目依赖：

```bash
pip install -r requirements.txt
```

## 基本使用

### 1. 启动 OmniParser 服务器

首先启动 OmniParser 服务器：

```bash
cd omnitool/omniparserserver
python omniparserserver.py --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05
```

默认服务器运行在 `localhost:8000`

### 2. 使用 CLI 工具

#### 方法一：直接运行 Python 脚本

```bash
cd omnitool
python omni_cli.py --input <输入内容> --server <服务器地址>
```

#### 方法二：使用便捷脚本

**Windows:**
```cmd
cd omnitool
omni-cli.bat --input <输入内容> --server <服务器地址>
```

**Linux/macOS:**
```bash
cd omnitool
./omni-cli.sh --input <输入内容> --server <服务器地址>
```

## 支持的输入类型

### 1. HTTP URL（自动下载图片）

```bash
python omni_cli.py --input "https://example.com/image.jpg" --server localhost:8000
```

### 2. 本地文件路径

```bash
python omni_cli.py --input "/path/to/your/image.jpg" --server localhost:8000
```

Windows 路径示例：
```cmd
python omni_cli.py --input "C:\Users\user\Desktop\image.jpg" --server localhost:8000
```

### 3. Base64 字符串

```bash
python omni_cli.py --input "iVBORw0KGgoAAAANSUhEUgAA..." --server localhost:8000 --input-type base64
```

## 命令行参数

| 参数 | 短参数 | 必需 | 默认值 | 说明 |
|------|--------|------|--------|------|
| `--input` | `-i` | 是 | - | 输入内容：HTTP URL、本地文件路径或 base64 字符串 |
| `--server` | `-s` | 否 | localhost:8000 | OmniParser 服务器地址 |
| `--input-type` | `-t` | 否 | auto | 输入类型：auto, url, file, base64 |
| `--output` | `-o` | 否 | - | 输出文件路径（可选，默认打印到控制台）|
| `--check-server` | - | 否 | - | 仅检查服务器状态 |
| `--timeout` | - | 否 | 30 | 下载超时时间（秒）|
| `--no-cleanup` | - | 否 | - | 不清理临时文件 |

## 使用示例

### 检查服务器状态

```bash
python omni_cli.py --check-server --server localhost:8000
```

### 解析网络图片并保存结果

```bash
python omni_cli.py --input "https://example.com/screenshot.png" --server localhost:8000 --output result.json
```

### 解析本地图片

```bash
python omni_cli.py --input "../imgs/example.jpg" --server localhost:8000
```

### 使用自定义服务器地址

```bash
python omni_cli.py --input "image.jpg" --server "http://192.168.1.100:8080"
```

### 指定输入类型（避免自动检测）

```bash
python omni_cli.py --input "very_long_base64_string" --input-type base64 --server localhost:8000
```

## 输出格式

CLI 工具会返回 JSON 格式的结果，包含：

- `som_image_base64`: 带标注的图像（base64 格式）
- `parsed_content_list`: 解析出的内容列表
- `latency`: 处理延迟（秒）

示例输出：
```json
{
  "som_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "parsed_content_list": [
    {
      "type": "button",
      "text": "Submit",
      "bbox": [100, 200, 150, 230]
    }
  ],
  "latency": 2.345
}
```

## 错误处理

工具会自动处理常见错误：
- 网络连接问题
- 文件不存在
- 服务器不可用
- 无效的 base64 字符串
- 超时错误

## 注意事项

1. 使用 HTTP URL 时，图片会临时下载到系统临时目录，处理完成后自动清理
2. 大图片可能需要较长处理时间，请耐心等待
3. 确保服务器正在运行且可访问
4. base64 字符串应该是完整的图片编码，不包含数据URL前缀（如 `data:image/jpeg;base64,`）

## 故障排除

### 服务器连接失败
1. 检查服务器是否正在运行：`python omni_cli.py --check-server`
2. 确认服务器地址和端口正确
3. 检查防火墙设置

### 图片下载失败
1. 检查网络连接
2. 确认图片 URL 有效且可访问
3. 尝试增加超时时间：`--timeout 60`

### 内存不足
1. 确保图片大小合理
2. 检查系统可用内存
3. 尝试使用较小的图片进行测试 
#!/usr/bin/env python3
"""
Omni CLI - Command line interface for OmniParser
支持三种输入方式：
1. HTTP URL - 自动下载图片并转换为 base64
2. 本地路径 - 读取本地图片文件并转换为 base64  
3. Base64 字符串 - 直接发送给服务器

使用示例:
python omni_cli.py --input "https://example.com/image.jpg" --server http://localhost:8000
python omni_cli.py --input "/path/to/image.jpg" --server http://localhost:8000
python omni_cli.py --input "base64_string_here" --server http://localhost:8000 --input-type base64
"""

import argparse
import requests
import base64
import os
import tempfile
import sys
import json
from pathlib import Path
from urllib.parse import urlparse
import mimetypes


def is_url(string):
    """检查字符串是否为有效的 URL"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False


def is_base64(string):
    """检查字符串是否为有效的 base64"""
    try:
        if isinstance(string, str):
            string = string.encode('ascii')
        decoded = base64.b64decode(string, validate=True)
        return True
    except Exception:
        return False


def download_image(url, timeout=30):
    """从 URL 下载图片到临时目录"""
    try:
        print(f"正在下载图片: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 获取文件扩展名
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            print(f"警告: Content-Type 可能不是图片格式: {content_type}")
        
        # 创建临时文件
        suffix = None
        if content_type:
            extension = mimetypes.guess_extension(content_type)
            if extension:
                suffix = extension
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        print(f"图片已下载到: {tmp_path}")
        return tmp_path
        
    except requests.RequestException as e:
        print(f"下载图片失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"处理下载时出错: {e}")
        sys.exit(1)


def image_to_base64(image_path):
    """将图片文件转换为 base64 字符串"""
    try:
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在: {image_path}")
            sys.exit(1)
        
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        print(f"已将图片转换为 base64: {len(encoded_string)} 字符")
        return encoded_string
        
    except Exception as e:
        print(f"转换图片为 base64 失败: {e}")
        sys.exit(1)


def send_parse_request(server_url, base64_image):
    """发送解析请求到服务器"""
    try:
        # 确保服务器 URL 格式正确
        if not server_url.startswith(('http://', 'https://')):
            server_url = f"http://{server_url}"
        
        parse_url = f"{server_url.rstrip('/')}/parse/"
        
        payload = {
            "base64_image": base64_image
        }
        
        print(f"正在发送请求到: {parse_url}")
        print("请求负载大小:", len(json.dumps(payload)), "字节")
        
        response = requests.post(
            parse_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=120  # 增加超时时间，因为解析可能需要较长时间
        )
        
        response.raise_for_status()
        return response.json()
        
    except requests.RequestException as e:
        print(f"发送请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应状态码: {e.response.status_code}")
            try:
                print(f"响应内容: {e.response.text}")
            except:
                pass
        sys.exit(1)
    except Exception as e:
        print(f"处理请求时出错: {e}")
        sys.exit(1)


def check_server_health(server_url):
    """检查服务器健康状态"""
    try:
        if not server_url.startswith(('http://', 'https://')):
            server_url = f"http://{server_url}"
        
        probe_url = f"{server_url.rstrip('/')}/probe/"
        
        print(f"检查服务器状态: {probe_url}")
        response = requests.get(probe_url, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print(f"服务器状态: {result.get('message', '未知')}")
        return True
        
    except Exception as e:
        print(f"服务器健康检查失败: {e}")
        return False


def save_result(result, output_file):
    """保存结果到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")


def cleanup_temp_file(temp_path):
    """清理临时文件"""
    try:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"已清理临时文件: {temp_path}")
    except Exception as e:
        print(f"清理临时文件失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Omni CLI - OmniParser 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --input "https://example.com/image.jpg" --server localhost:8000
  %(prog)s --input "/path/to/image.jpg" --server http://localhost:8000
  %(prog)s --input "base64_string" --server localhost:8000 --input-type base64
  %(prog)s --input "/path/to/image.jpg" --server localhost:8000 --output result.json
  %(prog)s --check-server --server localhost:8000
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=False,  # 改为不必需，因为 --check-server 不需要输入
        help='输入内容：HTTP URL、本地文件路径或 base64 字符串'
    )
    
    parser.add_argument(
        '--server', '-s',
        default='localhost:9333',
        help='OmniParser 服务器地址 (默认: localhost:8000)'
    )
    
    parser.add_argument(
        '--input-type', '-t',
        choices=['auto', 'url', 'file', 'base64'],
        default='auto',
        help='输入类型 (默认: auto - 自动检测)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='输出文件路径 (可选，默认打印到控制台)'
    )
    
    parser.add_argument(
        '--check-server',
        action='store_true',
        help='仅检查服务器状态'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='下载超时时间（秒）(默认: 30)'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='不清理临时文件'
    )
    
    args = parser.parse_args()
    
    # 仅检查服务器状态
    if args.check_server:
        if check_server_health(args.server):
            print("服务器运行正常")
            sys.exit(0)
        else:
            print("服务器不可用")
            sys.exit(1)
    
    # 如果没有输入且不是检查服务器，则报错
    if not args.input:
        print("错误: 需要提供 --input 参数或使用 --check-server")
        parser.print_help()
        sys.exit(1)
    
    # 检查服务器健康状态
    if not check_server_health(args.server):
        print("警告: 服务器似乎不可用，但继续尝试...")
    
    # 确定输入类型
    input_type = args.input_type
    if input_type == 'auto':
        if is_url(args.input):
            input_type = 'url'
        elif os.path.exists(args.input):
            input_type = 'file'
        elif is_base64(args.input):
            input_type = 'base64'
        else:
            print("错误: 无法自动识别输入类型，请使用 --input-type 指定")
            sys.exit(1)
    
    print(f"检测到输入类型: {input_type}")
    
    # 处理输入获取 base64
    base64_image = None
    temp_file_path = None
    
    try:
        if input_type == 'url':
            temp_file_path = download_image(args.input, timeout=args.timeout)
            base64_image = image_to_base64(temp_file_path)
            
        elif input_type == 'file':
            base64_image = image_to_base64(args.input)
            
        elif input_type == 'base64':
            base64_image = args.input
            print("使用提供的 base64 字符串")
        
        # 发送解析请求
        print("\n开始解析...")
        result = send_parse_request(args.server, base64_image)
        
        # 处理结果
        print(f"\n解析完成！延迟: {result.get('latency', 'N/A')} 秒")
        print(f"解析内容项数: {len(result.get('parsed_content_list', []))}")
        
        if args.output:
            save_result(result, args.output)
        else:
            print("\n解析结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    finally:
        # 清理临时文件
        if temp_file_path and not args.no_cleanup:
            cleanup_temp_file(temp_file_path)


if __name__ == "__main__":
    main() 



'''

python misc/omniparser_tools.py --input imgs/live.png   --server 172.27.1.113:9333 -t file            
python misc/omniparser_tools.py --input imgs/live.png   --server 100.122.57.128:9333 -t file            
'''
#!/usr/bin/env python3
"""
药品说明书图片预处理脚本
功能：
1. 统一转换为JPG格式
2. 自动旋转校正
3. 调整分辨率
4. 批量重命名
"""

import os
import argparse
from PIL import Image, ImageOps
from pathlib import Path

def auto_rotate(image):
    """通过EXIF信息自动旋转图片"""
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image

def process_image(input_path, output_dir, target_size=None, quality=85):
    """
    处理单张图片
    :param input_path: 输入图片路径
    :param output_dir: 输出目录
    :param target_size: (width, height) 目标尺寸
    :param quality: JPG质量 (1-100)
    """
    try:
        # 读取图片并自动旋转
        img = Image.open(input_path)
        img = auto_rotate(img)
        
        # 转换为RGB模式（处理PNG透明背景）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 调整尺寸（保持宽高比）
        if target_size:
            img.thumbnail(target_size, Image.LANCZOS)
        
        # 生成输出路径
        stem = Path(input_path).stem
        output_path = Path(output_dir) / f"{stem}.jpg"
        
        # 保存为JPG
        img.save(output_path, "JPEG", quality=quality, optimize=True)
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def batch_process(input_dir, output_dir, target_size=None, quality=85):
    """
    批量处理目录中的所有图片
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的输入格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 遍历处理
    processed = 0
    for file in Path(input_dir).iterdir():
        if file.suffix.lower() in valid_exts:
            process_image(file, output_dir, target_size, quality)
            processed += 1
    
    print(f"\nDone. Processed {processed} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="药品说明书图片预处理工具")
    parser.add_argument("--input", required=True, help="输入图片目录路径")
    parser.add_argument("--output", required=True, help="输出目录路径")
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="目标尺寸，如 1200 1800")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPG质量 (1-100), 默认85")
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("\n" + "="*50)
    print(f"Input Directory: {args.input}")
    print(f"Output Directory: {args.output}")
    print(f"Target Size: {args.size if args.size else '保持原尺寸'}")
    print(f"JPG Quality: {args.quality}")
    print("="*50 + "\n")
    
    # 运行批处理
    batch_process(
        input_dir=args.input,
        output_dir=args.output,
        target_size=args.size,
        quality=args.quality
    )

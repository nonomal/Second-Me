#!/usr/bin/env python3
"""
跨平台兼容工具：将带有反斜杠(\)续行的多行shell命令转换为单行命令
用法: python convert_to_single_line.py 输入文件 [输出文件]
如果未指定输出文件，将覆盖输入文件
"""

import sys
import os
import re

def convert_multiline_to_single_line(file_path, output_path=None):
    """
    将多行命令（以反斜杠结尾）转换为单行命令
    """
    if output_path is None:
        output_path = file_path

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 保留原始文件的权限
        original_mode = os.stat(file_path).st_mode
            
        # 使用正则表达式找到以 \ 结尾的行并移除换行符和 \
        pattern = r'\\\s*\n\s*'
        converted_content = re.sub(pattern, ' ', content)
            
        # 写入转换后的内容到输出文件
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(converted_content)
            
        # 恢复原始文件的权限
        os.chmod(output_path, original_mode)
        
        print(f"已成功将 {file_path} 转换为单行命令格式，保存至 {output_path}")
        return True
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def convert_single_line_to_multiline(file_path, output_path=None, line_prefix="--"):
    """
    将单行命令转换回多行格式（使用 \ 续行符），每个参数单独一行
    主要用于将以 line_prefix 开头的参数分行
    """
    if output_path is None:
        output_path = file_path

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 保留原始文件的权限
        original_mode = os.stat(file_path).st_mode
        
        # 查找命令行并进行转换
        lines = content.split('\n')
        converted_lines = []
        
        for line in lines:
            if line_prefix in line:
                # 分割命令行
                parts = line.split(line_prefix)
                # 保存第一部分（命令名）
                first_part = parts[0].rstrip() + " \\"
                converted_lines.append(first_part)
                
                # 处理每个参数
                for i, part in enumerate(parts[1:], 1):
                    if part.strip():
                        # 如果不是最后一个参数，添加续行符
                        if i < len(parts) - 1:
                            converted_lines.append(line_prefix + part.rstrip() + " \\")
                        else:
                            converted_lines.append(line_prefix + part.rstrip())
            else:
                converted_lines.append(line)
        
        # 写入转换后的内容到输出文件
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(converted_lines))
        
        # 恢复原始文件的权限
        os.chmod(output_path, original_mode)
        
        print(f"已成功将 {file_path} 转换为多行命令格式，保存至 {output_path}")
        return True
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} 输入文件 [输出文件] [--to-multiline]")
        return
    
    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    
    # 检查是否需要转换为多行
    to_multiline = '--to-multiline' in sys.argv
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    if to_multiline:
        convert_single_line_to_multiline(file_path, output_path)
    else:
        convert_multiline_to_single_line(file_path, output_path)

if __name__ == "__main__":
    main()

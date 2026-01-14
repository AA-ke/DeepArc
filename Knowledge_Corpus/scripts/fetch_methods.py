"""
Markdown读取脚本
从core_papers_md目录读取Markdown文件，提取标题和Methods部分，组织成适合RAG的格式
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


def generate_doc_id(source: str, identifier: str) -> str:
    """生成唯一文档ID"""
    unique_string = f"{source}_{identifier}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:16]


def extract_title_from_markdown(content: str) -> str:
    """从Markdown文件中提取第一个一级标题（# 开头的）"""
    lines = content.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        # 查找第一个一级标题（# 开头，后面跟空格）
        if re.match(r'^#\s+', line_stripped):
            # 移除 # 和空格，返回标题文本
            title = re.sub(r'^#\s+', '', line_stripped).strip()
            if title:
                return title
    
    return ""


def find_methods_section_in_markdown(content: str) -> Optional[str]:
    """从Markdown文件中查找并提取Methods部分"""
    lines = content.split('\n')
    
    # Methods部分的可能标题变体（Markdown格式，以 # 开头，可能带编号/罗马数字前缀和冒号）
    # 支持示例：
    #   # Methods
    #   # 2. Methods
    #   # III. METHODS
    #   # 3 METHOD
    #   # 3 Design and implementations
    #   # 2. Approach
    #   # 3 HyenaDNA Long-Range Genomic Foundation Models
    method_patterns = [
        r'^#\s+(?:[0-9IVXLCDM]+\.?\s+)?'
        r'(?:Methods?|METHODS?|Methodology|METHODOLOGY|Experimental\s+Methods?|EXPERIMENTAL\s+METHODS?'
        r'|Materials\s+and\s+Methods?|MATERIALS\s+AND\s+METHODS?|Methods\s+and\s+Materials|METHODS\s+AND\s+MATERIALS'
        r'|Design\s+and\s+implementations?|DESIGN\s+AND\s+IMPLEMENTATIONS?'
        r'|Approach|APPROACH'
        r'|HyenaDNA\s+Long-Range\s+Genomic\s+Foundation\s+Models)'
        r'\s*:?\s*$',
    ]
    
    # 查找Methods部分的开始位置
    method_start = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # 检查是否是Methods标题（# Methods 或 # METHODS 等，支持冒号）
        for pattern in method_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                method_start = i
                break
        if method_start is not None:
            break
    
    if method_start is None:
        return None
    
    # 查找Methods部分的结束位置（下一个一级标题 # 开头）
    method_end = len(lines)
    
    # 从Methods部分开始后查找下一个一级标题
    for i in range(method_start + 1, len(lines)):
        line_stripped = lines[i].strip()
        # 检查是否是下一个一级标题（# 开头，但不是 Methods 的变体）
        if re.match(r'^#\s+', line_stripped):
            # 检查是否是其他主要章节（不是 Methods 的子标题）
            next_title = re.sub(r'^#\s+', '', line_stripped).strip()
            # 移除可能的冒号
            next_title = re.sub(r':\s*$', '', next_title).strip()
            # 排除 Methods 的变体（可能是子标题）
            if not re.match(r'(?i)^(?:Methods?|METHODS?|Methodology|METHODOLOGY)', next_title):
                # 检查是否是主要章节（Discussion, References, Data availability等）
                if re.match(r'(?i)^(?:Discussion|References?|Data\s+availability|Code\s+availability|Acknowledgments?|Author\s+contributions|Competing\s+interests|Additional\s+information|Reporting\s+Summary|Statistics|Software\s+and\s+code|Field-specific\s+reporting)', next_title):
                    method_end = i
                    break
    
    # 提取Methods部分内容（包含标题行）
    methods_lines = lines[method_start:method_end]
    methods_text = '\n'.join(methods_lines).strip()
    
    # 移除Methods标题行（保留内容，支持数字/罗马数字前缀和冒号）
    methods_text = re.sub(
        r'^#\s+(?:[0-9IVXLCDM]+\.?\s+)?'
        r'(?:Methods?|METHODS?|Methodology|METHODOLOGY|Experimental\s+Methods?|EXPERIMENTAL\s+METHODS?'
        r'|Materials\s+and\s+Methods?|MATERIALS\s+AND\s+METHODS?|Methods\s+and\s+Materials|METHODS\s+AND\s+MATERIALS'
        r'|Design\s+and\s+implementations?|DESIGN\s+AND\s+IMPLEMENTATIONS?'
        r'|Approach|APPROACH'
        r'|HyenaDNA\s+Long-Range\s+Genomic\s+Foundation\s+Models)'
        r'\s*:?\s*$',
        '',
        methods_text,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    
    # 清理文本：移除过多的空白行，保留段落结构
    methods_text = re.sub(r'\n{3,}', '\n\n', methods_text)
    
    # 确保有足够的内容（至少100个字符）
    if len(methods_text.strip()) < 100:
        return None
    
    return methods_text if methods_text else None


def preserve_paragraphs(text: str) -> str:
    """保留段落结构，确保段落之间用双换行符分隔"""
    # 清理文本
    text = text.strip()
    
    # 将多个连续换行符统一为双换行符（段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 确保段落之间是双换行符
    # 单换行符通常表示同一段落内的换行，双换行符表示段落分隔
    lines = text.split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # 空行表示段落结束
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
        else:
            current_para.append(line)
    
    # 添加最后一个段落
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # 用双换行符连接段落
    return '\n\n'.join(paragraphs)


def process_markdown(md_path: Path) -> Optional[Dict]:
    """处理单个Markdown文件，提取标题和Methods部分"""
    print(f"  处理: {md_path.name}", flush=True)
    
    # 读取Markdown文件
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ⚠️ 无法读取文件: {md_path.name}, 错误: {e}", flush=True)
        return None
    
    if not content.strip():
        print(f"  ⚠️ 文件为空: {md_path.name}", flush=True)
        return None
    
    # 提取标题（第一个 # 标题）
    title = extract_title_from_markdown(content)
    if not title:
        # 如果无法提取标题，使用文件名（去除扩展名和前缀）
        title = md_path.stem
        # 移除可能的文件名前缀（如 MinerU_markdown_）
        title = re.sub(r'^MinerU_markdown_', '', title)
        title = title.replace('_', ' ').replace('-', ' ')
        print(f"  ⚠️ 无法提取标题，使用文件名: {title}", flush=True)
    
    # 提取Methods部分
    methods_text = find_methods_section_in_markdown(content)
    has_methods = False
    
    if methods_text:
        # 保留段落结构（但保留Markdown格式，不转换为纯文本）
        # 只清理多余的空白行
        methods_text = re.sub(r'\n{3,}', '\n\n', methods_text.strip())
        has_methods = True
    else:
        print(f"  ⚠️ 未找到Methods部分: {md_path.name}，methods字段留空供手动填写", flush=True)
        methods_text = ""  # 留空，供用户手动填写
    
    # 生成文档ID（使用文件名，去除扩展名）
    source_id = md_path.stem
    doc_id = generate_doc_id("core_papers_md", source_id)
    
    # 组织成统一格式
    doc = {
        "doc_id": doc_id,
        "source": "Core Papers MD",
        "source_id": source_id,
        "title": title,
        "abstract": "",  # 不提取abstract
        "authors": "",
        "journal": "",
        "date": "",
        "doi": "",
        "url": "",
        "keywords": [],
        "full_text": "",  # 不存储全文，只存储Methods部分
        "methods": methods_text,  # Methods部分，保留Markdown格式和段落结构（如果找到）
        "metadata": {
            "md_filename": md_path.name,
            "md_path": str(md_path),
            "has_methods": has_methods  # 标记是否找到Methods部分
        }
    }
    
    if has_methods:
        print(f"  ✓ 提取成功: {title[:50]}...", flush=True)
    else:
        print(f"  ⚠️ 已保存（Methods待填写）: {title[:50]}...", flush=True)
    
    return doc


def main():
    """主函数：处理core_papers_md目录中的所有Markdown文件"""
    print("="*80)
    print("📄 Markdown读取脚本 - 提取标题和Methods部分")
    print("="*80)
    
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    core_papers_md_dir = project_root / "Knowledge_Corpus" / "core_papers_md"
    output_dir = project_root / "Knowledge_Corpus" / "data" / "raw"
    
    if not core_papers_md_dir.exists():
        print(f"❌ 目录不存在: {core_papers_md_dir}", flush=True)
        sys.exit(1)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有Markdown文件
    md_files = list(core_papers_md_dir.glob("*.md"))
    if not md_files:
        print(f"❌ 未找到Markdown文件: {core_papers_md_dir}", flush=True)
        sys.exit(1)
    
    print(f"\n📚 找到 {len(md_files)} 个Markdown文件", flush=True)
    print(f"📂 输出目录: {output_dir}\n", flush=True)
    
    # 处理每个Markdown文件
    all_docs = []
    success_count = 0
    with_methods_count = 0
    without_methods_count = 0
    failed_count = 0
    
    for md_path in sorted(md_files):
        try:
            doc = process_markdown(md_path)
            if doc:
                all_docs.append(doc)
                success_count += 1
                # 统计有Methods和没有Methods的数量
                if doc.get("metadata", {}).get("has_methods", False):
                    with_methods_count += 1
                else:
                    without_methods_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  ❌ 处理失败 {md_path.name}: {e}", flush=True)
            failed_count += 1
    
    # 保存结果
    if all_docs:
        output_file = output_dir / "core_papers_md_methods.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ 处理完成!")
        print(f"  成功: {success_count} 个文件")
        print(f"    - 包含Methods: {with_methods_count} 个")
        print(f"    - Methods待填写: {without_methods_count} 个")
        print(f"  失败: {failed_count} 个文件")
        print(f"  输出文件: {output_file}")
        print(f"\n💡 提示: Methods字段为空的文档可以手动填写Methods内容", flush=True)
        print(f"{'='*80}", flush=True)
    else:
        print(f"\n❌ 没有成功提取任何文档", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

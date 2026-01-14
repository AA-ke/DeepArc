"""
QA Pairs 生成脚本
从 documents_deduped.json 读取文档，使用 LLM 生成高质量的 QA pairs
"""

import json
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Knowledge_Corpus/scripts -> Knowledge_Corpus -> RE-Agent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_settings


# 配置参数
BATCH_SIZE = 10  # 每批处理的文档数量
DELAY_BETWEEN_BATCHES = 2.0  # 批次之间的延迟（秒）
DELAY_BETWEEN_DOCS = 0.5  # 文档之间的延迟（秒）
MAX_QA_PAIRS_PER_DOC = 2  # 每个文档生成的最大 QA pairs 数量

# 输入输出路径
INPUT_FILE = Path(__file__).parent.parent / "data" / "cleaned" / "documents_deduped.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "qa_pairs.json"
PROGRESS_FILE = Path(__file__).parent.parent / "data" / "qa_pairs_progress.json"


def load_documents(input_file: Path) -> List[Dict[str, Any]]:
    """加载文档数据"""
    print(f"📖 正在加载文档: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"✓ 已加载 {len(documents)} 个文档")
    return documents


def load_progress() -> Dict[str, Any]:
    """加载进度信息"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed_doc_ids": [],
        "last_processed_index": -1,
        "total_qa_pairs": 0,
        "start_time": datetime.now().isoformat()
    }


def save_progress(progress: Dict[str, Any]):
    """保存进度信息"""
    progress["last_update"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_existing_qa_pairs(output_file: Path) -> List[Dict[str, Any]]:
    """加载已存在的 QA pairs"""
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️ 加载已有 QA pairs 失败: {e}，将创建新文件")
    return []


def save_qa_pairs(qa_pairs: List[Dict[str, Any]], output_file: Path, append: bool = False):
    """保存 QA pairs"""
    if append:
        existing = load_existing_qa_pairs(output_file)
        qa_pairs = existing + qa_pairs
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)


def create_qa_prompt(doc: Dict[str, Any]) -> str:
    """Create prompt for generating QA pairs"""
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    authors = doc.get("authors", "")
    journal = doc.get("journal", "")
    
    prompt = f"""Based on the following scientific literature, generate {MAX_QA_PAIRS_PER_DOC} high-quality question-answer pairs (QA pairs).

**Literature Information:**
- Title: {title}
- Authors: {authors}
- Journal: {journal}
- Abstract: {abstract}

**CRITICAL REQUIREMENTS FOR QUESTIONS:**
Questions MUST be:
1. **Conceptual**: Focus on fundamental concepts, principles, and theoretical frameworks rather than specific experimental details
2. **Generalizable**: Address broad, transferable knowledge that applies beyond this specific study
3. **Universal**: Cover principles and methods that are relevant across different contexts or domains
4. **Methodological**: Emphasize research approaches, techniques, and methodological insights
5. **Factual**: Based on established facts and findings, but at a conceptual level

**AVOID:**
- Questions about specific numerical values, percentages, or exact measurements
- Questions about particular gene names, protein names, or specific biological entities (unless asking about the general concept)
- Questions about specific experimental conditions, sample sizes, or detailed procedures
- Questions about specific dates, locations, or study-specific details
- Questions that require memorizing exact quotes or specific phrases from the abstract

**PREFERRED QUESTION TYPES:**
1. **Conceptual questions**: "What is the fundamental principle/concept of...?"
2. **Methodological questions**: "What approach/method/model is used to...?" or "How does the methodology address...?"
3. **Generalizable questions**: "What are the key factors/mechanisms that...?" or "What general insights can be drawn about...?"
4. **Comparative questions**: "How does this approach compare to other methods in terms of...?" (focus on general principles, not specific comparisons)
5. **Application questions**: "What are the general applications/implications of...?" (at a conceptual level)

**Requirements:**
1. Generate {MAX_QA_PAIRS_PER_DOC} question-answer pairs that focus on conceptual, generalizable, and methodological aspects
2. Questions should be:
   - Conceptually oriented and broadly applicable
   - Focused on principles, methods, and general insights
   - Answerable from the abstract but at a high level of abstraction
   - Avoiding article-specific details

3. Answers should be:
   - Accurate, complete, and based on the literature content
   - Concise and clear (typically 2-4 sentences)
   - Emphasizing conceptual understanding and general principles
   - Including key methodological or theoretical insights

4. The output format must be a strict JSON array, with each element containing "question" and "answer" fields:
```json
[
  {{
    "question": "Question 1",
    "answer": "Answer 1"
  }},
  {{
    "question": "Question 2",
    "answer": "Answer 2"
  }}
]
```

Please output the JSON array directly without any additional text or explanation."""
    
    return prompt


async def generate_qa_pairs_for_doc(
    llm: ChatOpenAI,
    doc: Dict[str, Any],
    doc_index: int,
    total_docs: int,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """为单个文档生成 QA pairs（带重试机制）"""
    prompt = create_qa_prompt(doc)
    
    messages = [
        SystemMessage(content="You are a professional scientific literature analyst expert, skilled at extracting key information from literature and generating high-quality question-answer pairs."),
        HumanMessage(content=prompt)
    ]
    
    print(f"  [{doc_index + 1}/{total_docs}] 正在为文档生成 QA pairs: {doc.get('title', 'N/A')[:60]}...")
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # API 调用
            response = await llm.ainvoke(messages)
            content = response.content.strip()
            
            # 尝试提取 JSON
            # 移除可能的代码块标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # 解析 JSON
            try:
                qa_pairs = json.loads(content)
                if not isinstance(qa_pairs, list):
                    qa_pairs = [qa_pairs]
                
                # 验证格式并添加元数据
                validated_pairs = []
                for qa in qa_pairs:
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                        validated_pairs.append({
                            "question": qa["question"],
                            "answer": qa["answer"],
                            "doc_id": doc.get("doc_id", ""),
                            "doc_title": doc.get("title", ""),
                            "doc_source": doc.get("source", ""),
                            "doc_source_id": doc.get("source_id", ""),
                            "generated_at": datetime.now().isoformat()
                        })
                
                if validated_pairs:
                    print(f"    ✓ 成功生成 {len(validated_pairs)} 个 QA pairs")
                    return validated_pairs
                else:
                    print(f"    ⚠️ 未生成有效的 QA pairs")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return []
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"    ⚠️ JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"    Raw content (first 500 chars): {content[:500]}")
                    await asyncio.sleep(2)  # 等待后重试
                    continue
                else:
                    print(f"    ⚠️ JSON 解析失败（已重试 {max_retries} 次）: {e}")
                    print(f"    Raw content (first 500 chars): {content[:500]}")
                    return []
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️ API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                print(f"    ❌ 生成 QA pairs 失败（已重试 {max_retries} 次）: {e}")
                import traceback
                traceback.print_exc()
                return []
    
    return []  # 所有重试都失败


async def process_batch(
    llm: ChatOpenAI,
    documents: List[Dict[str, Any]],
    batch_start: int,
    batch_end: int,
    progress: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """处理一批文档"""
    batch = documents[batch_start:batch_end]
    all_qa_pairs = []
    
    for i, doc in enumerate(batch):
        doc_index = batch_start + i
        doc_id = doc.get("doc_id", "")
        
        # 跳过已处理的文档
        if doc_id in progress["processed_doc_ids"]:
            print(f"  [{doc_index + 1}/{len(documents)}] 跳过已处理文档: {doc_id}")
            continue
        
        # 生成 QA pairs
        qa_pairs = await generate_qa_pairs_for_doc(llm, doc, doc_index, len(documents))
        
        if qa_pairs:
            all_qa_pairs.extend(qa_pairs)
            progress["processed_doc_ids"].append(doc_id)
            progress["total_qa_pairs"] += len(qa_pairs)
        
        # 文档间延迟
        if i < len(batch) - 1:
            await asyncio.sleep(DELAY_BETWEEN_DOCS)
    
    progress["last_processed_index"] = batch_end - 1
    return all_qa_pairs


async def main():
    """主函数"""
    print("=" * 80)
    print("QA Pairs 生成脚本")
    print("=" * 80)
    print()
    
    # 加载设置
    settings = get_settings()
    if not settings.openai_api_key:
        print("❌ 错误: 未找到 OPENAI_API_KEY，请在 .env 文件中设置")
        return
    
    # 初始化 LLM
    model_name = "gpt-4o-mini"
    print(f"🤖 初始化 LLM: {model_name}")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,  # 稍微提高温度以增加多样性
        openai_api_key=settings.openai_api_key
    )
    print("✓ LLM 初始化完成")
    print()
    
    # 加载文档
    if not INPUT_FILE.exists():
        print(f"❌ 错误: 输入文件不存在: {INPUT_FILE}")
        return
    
    documents = load_documents(INPUT_FILE)
    
    # 加载进度
    progress = load_progress()
    print(f"📊 进度信息:")
    print(f"  - 已处理文档: {len(progress['processed_doc_ids'])}/{len(documents)}")
    print(f"  - 已生成 QA pairs: {progress['total_qa_pairs']}")
    print(f"  - 上次处理索引: {progress.get('last_processed_index', -1)}")
    print()
    
    # 确定起始位置
    start_index = progress.get("last_processed_index", -1) + 1
    if start_index >= len(documents):
        print("✓ 所有文档已处理完成！")
        return
    
    print(f"🚀 开始处理，从索引 {start_index} 开始")
    print(f"📦 批次大小: {BATCH_SIZE}, 批次延迟: {DELAY_BETWEEN_BATCHES}秒, 文档延迟: {DELAY_BETWEEN_DOCS}秒")
    print()
    
    # 分批处理
    all_qa_pairs = []
    total_batches = (len(documents) - start_index + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        batch_start = start_index + batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(documents))
        
        print(f"📦 处理批次 {batch_num + 1}/{total_batches} (文档 {batch_start + 1}-{batch_end})")
        
        # 处理批次
        batch_qa_pairs = await process_batch(llm, documents, batch_start, batch_end, progress)
        all_qa_pairs.extend(batch_qa_pairs)
        
        # 保存进度和结果
        save_progress(progress)
        if batch_qa_pairs:
            # 增量保存：加载已有数据，追加新数据，然后保存
            existing_qa_pairs = load_existing_qa_pairs(OUTPUT_FILE)
            all_qa_pairs_to_save = existing_qa_pairs + batch_qa_pairs
            save_qa_pairs(all_qa_pairs_to_save, OUTPUT_FILE, append=False)
            print(f"  ✓ 已保存 {len(batch_qa_pairs)} 个新 QA pairs（总计: {len(all_qa_pairs_to_save)}）")
        
        print(f"  📊 累计: {len(all_qa_pairs)} 个 QA pairs, {len(progress['processed_doc_ids'])} 个文档已处理")
        print()
        
        # 批次间延迟（最后一个批次不需要延迟）
        if batch_num < total_batches - 1:
            print(f"  ⏳ 等待 {DELAY_BETWEEN_BATCHES} 秒...")
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)
    
    # 最终统计
    print("=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"📊 最终统计:")
    print(f"  - 处理文档数: {len(progress['processed_doc_ids'])}/{len(documents)}")
    print(f"  - 生成 QA pairs: {progress['total_qa_pairs']}")
    print(f"  - 输出文件: {OUTPUT_FILE}")
    print(f"  - 进度文件: {PROGRESS_FILE}")
    print()


if __name__ == "__main__":
    asyncio.run(main())


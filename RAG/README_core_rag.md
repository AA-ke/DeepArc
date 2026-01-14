# Core RAG 系统使用说明

## 概述

`CoreRAGSystem` 是专门用于处理核心论文 Methods 部分的 RAG 系统，从 `core_papers_md_methods.json` 加载知识库。

## 主要特性

1. **按段落分块**：使用双换行符（`\n\n`）分割段落，保持语义完整性
2. **保留标题**：每个 chunk 都包含其对应的 Markdown 标题（如 `# 4. Caduceus`）
3. **智能分块**：如果段落过长，会进一步按句子分割，但仍保留标题
4. **重叠策略**：chunk 之间保留重叠以维持上下文连续性

## 使用方法

### 1. 初始化系统

```python
from RAG.core_rag import CoreRAGSystem

# 使用默认配置
rag = CoreRAGSystem()

# 或使用自定义配置
config = {
    "rag_top_k": 10,
    "chunk_size": 2048,
    "enable_reranking": True
}
rag = CoreRAGSystem(config=config)
```

### 2. 加载知识库

```python
from pathlib import Path

# 使用默认路径
rag.load_knowledge_base()

# 或指定自定义路径
knowledge_base_path = Path("path/to/core_papers_md_methods.json")
rag.load_knowledge_base(file_path=knowledge_base_path)
```

### 3. 执行检索

```python
import asyncio

async def query_example():
    rag = CoreRAGSystem()
    rag.load_knowledge_base()
    
    # 执行检索
    result = await rag.retrieve(
        query="DNA sequence modeling architecture",
        top_k=5
    )
    
    # 查看结果
    for res in result.results:
        print(f"分数: {res.score:.4f}")
        print(f"标题: {res.metadata['title']}")
        print(f"章节: {res.metadata['section_title']}")
        print(f"内容: {res.content[:200]}...")
        print("-" * 60)

asyncio.run(query_example())
```

### 4. 获取统计信息

```python
stats = rag.get_collection_stats()
print(f"文档数量: {stats['document_count']}")
```

## 分块策略详解

### 段落分割

- 使用 `\n\n`（双换行符）作为段落分隔符
- 每个段落作为一个独立的语义单元

### 标题提取

- 从 Methods 文本中提取 Markdown 标题（以 `#` 开头）
- 每个 chunk 都会包含其对应的 section 标题
- 如果段落前没有 section 标题，则使用文档标题

### 长段落处理

- 如果段落长度超过 `chunk_size`（默认 2048 字符），会进一步分割
- 优先按句子分割（句号、问号、感叹号）
- 确保每个 chunk 不超过大小限制

### 重叠策略

- chunk 之间保留 `chunk_overlap`（默认 200 字符）的重叠
- 重叠部分从前一个 chunk 的末尾提取
- 尽量在单词边界截取，保持语义完整

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding 模型 |
| `rag_top_k` | 5 | 返回结果数量 |
| `rag_similarity_threshold` | 0.3 | 相似度阈值 |
| `chunk_size` | 2048 | 每个 chunk 的最大字符数 |
| `chunk_overlap` | 200 | chunk 之间的重叠字符数 |
| `enable_reranking` | True | 是否启用重排 |
| `rerank_top_k` | 10 | 重排候选数量 |
| `final_top_k` | 5 | 最终返回结果数 |

## 数据格式

输入 JSON 文件格式：

```json
[
  {
    "doc_id": "xxx",
    "title": "论文标题",
    "methods": "# 4. Section Title\n\n段落内容...",
    "metadata": {
      "has_methods": true
    }
  }
]
```

## 注意事项

1. 只有 `methods` 字段非空的文档才会被索引
2. 每个 chunk 的 metadata 中包含 `section_title` 字段，表示该 chunk 所属的 section
3. 检索结果按相关性分数排序，分数越高表示越相关


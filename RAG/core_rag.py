"""
RAG/core_rag.py
Core Papers Methods RAG系统 - 专门处理核心论文的Methods部分
从 Knowledge_Corpus/data/raw/core_papers_md_methods.json 加载知识库
分块策略：按段落分块，保留标题
"""

import time
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("⚠️ CrossEncoder未安装，将使用简单重排。安装: pip install sentence-transformers")


# ==================== 数据模型定义 ====================

@dataclass
class RetrievalResult:
    """检索结果"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str


@dataclass
class RAGQueryResult:
    """RAG查询结果"""
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    retrieval_time_ms: float = 0.0


# ==================== 配置类 ====================

class CoreRAGSettings:
    """Core RAG系统配置"""
    def __init__(self):
        # Embedding模型
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # 向量数据库配置
        self.vector_db_type = "chromadb"
        self.chroma_persist_directory = "./RAG/chroma_db_core"
        
        # 检索配置
        self.rag_top_k = 5
        self.rag_similarity_threshold = 0.3
        
        # 文档分块配置
        self.chunk_size = 2048  # 每个chunk的字符数（Methods部分通常较长）
        self.chunk_overlap = 200  # chunk之间的重叠字符数
        
        # 重排配置
        self.enable_reranking = True
        self.rerank_top_k = 10
        self.final_top_k = 5
        self.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # 知识库路径
        self.knowledge_base_path = Path(__file__).parent.parent / "Knowledge_Corpus" / "data" / "raw" / "core_papers_md_methods.json"


# ==================== Core RAG系统 ====================

class CoreRAGSystem:
    """
    Core Papers Methods RAG系统
    
    专门处理核心论文的Methods部分，按段落分块并保留标题
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Core RAG系统
        
        Args:
            config: 可选的配置字典，覆盖默认配置
        """
        self.settings = CoreRAGSettings()
        if config:
            for key, value in config.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
        
        # 初始化embedding模型
        print(f"正在加载embedding模型: {self.settings.embedding_model}...")
        self.embedding_model = SentenceTransformer(self.settings.embedding_model)
        print(f"✓ 加载embedding模型完成")
        
        # 初始化cross-encoder模型（用于重排）
        self.cross_encoder = None
        if self.settings.enable_reranking and CROSS_ENCODER_AVAILABLE:
            try:
                print(f"正在加载cross-encoder模型: {self.settings.cross_encoder_model}...")
                self.cross_encoder = CrossEncoder(self.settings.cross_encoder_model)
                print(f"✓ 加载cross-encoder模型完成")
            except Exception as e:
                print(f"⚠️ Cross-encoder模型加载失败: {e}，将使用简单重排")
                self.cross_encoder = None
        elif self.settings.enable_reranking and not CROSS_ENCODER_AVAILABLE:
            print("⚠️ CrossEncoder未安装，将使用简单重排。安装: pip install sentence-transformers")
        
        # 初始化向量数据库
        self._init_vector_db()
        
        # 初始化集合
        self.collection = None
        self._initialize_collection()
    
    def _init_vector_db(self):
        """初始化向量数据库客户端"""
        if self.settings.vector_db_type == "chromadb":
            # 确保目录存在
            persist_dir = Path(self.settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir)
            )
            print(f"✓ 初始化ChromaDB: {persist_dir}")
        else:
            raise ValueError(f"不支持的向量数据库类型: {self.settings.vector_db_type}")
    
    def _initialize_collection(self):
        """初始化向量数据库集合"""
        print("初始化向量数据库集合...")
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="core_papers_methods",
                metadata={
                    "description": "核心论文Methods部分知识库",
                    "source": "core_papers_md_methods.json"
                }
            )
            print(f"  ✓ core_papers_methods")
        except Exception as e:
            print(f"  ✗ core_papers_methods: {e}")
            raise
    
    def _extract_section_title(self, text: str, start_pos: int) -> str:
        """
        从文本的指定位置向前查找最近的Markdown标题
        
        Args:
            text: 完整文本
            start_pos: 当前段落开始位置
        
        Returns:
            找到的标题（不含#），如果没找到返回空字符串
        """
        # 向前查找最近的标题（以 # 开头）
        # 查找从文档开始到start_pos之间的最后一个标题
        text_before = text[:start_pos]
        
        # 匹配所有Markdown标题（# 开头，后面跟空格和标题文本）
        title_pattern = r'^#+\s+(.+)$'
        matches = list(re.finditer(title_pattern, text_before, re.MULTILINE))
        
        if matches:
            # 返回最后一个匹配的标题（最接近当前段落）
            last_match = matches[-1]
            title = last_match.group(1).strip()
            return title
        
        return ""
    
    def _chunk_by_paragraphs(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """
        按段落分块，保留标题
        
        策略：
        1. 按双换行符（\n\n）分割段落
        2. 每个段落块保留其对应的Markdown标题
        3. 如果段落太长，进一步分割但保留标题
        
        Args:
            text: 要分块的文本（methods部分）
            title: 文档标题
        
        Returns:
            块列表，每个块包含：
            - content: 块内容（包含标题）
            - section_title: 该块所属的section标题
            - chunk_index: 块索引
        """
        if not text or not text.strip():
            return []
        
        # 按双换行符分割段落
        paragraphs = []
        for para in text.split('\n\n'):
            para = para.strip()
            if para:
                paragraphs.append(para)
        
        if not paragraphs:
            return []
        
        # 重新组合文本以查找标题位置
        full_text = '\n\n'.join(paragraphs)
        
        chunks = []
        current_pos = 0
        
        for para_idx, para in enumerate(paragraphs):
            # 查找该段落对应的section标题
            para_start = full_text.find(para, current_pos)
            section_title = self._extract_section_title(full_text, para_start)
            
            # 如果段落太长，进一步分割
            if len(para) <= self.settings.chunk_size:
                # 段落长度合适，直接作为一个chunk
                # 始终保留文档标题作为主标题
                if title:
                    if section_title:
                        # 同时包含文档标题和section标题
                        chunk_content = f"# {title}\n\n## {section_title}\n\n{para}"
                    else:
                        # 只有文档标题
                        chunk_content = f"# {title}\n\n{para}"
                else:
                    # 如果没有文档标题，使用section标题
                    chunk_content = f"## {section_title}\n\n{para}" if section_title else para
                
                chunks.append({
                    "content": chunk_content,
                    "section_title": section_title or "",
                    "chunk_index": para_idx,
                    "original_para": para
                })
            else:
                # 段落太长，需要进一步分割
                # 按句子分割（尽量保持语义完整）
                sentences = re.split(r'([.!?]\s+)', para)
                current_chunk = ""
                chunk_sentences = []
                
                for i in range(0, len(sentences), 2):
                    if i + 1 < len(sentences):
                        sentence = sentences[i] + sentences[i + 1]
                    else:
                        sentence = sentences[i]
                    
                    # 如果当前chunk加上新句子不超过限制，合并
                    if len(current_chunk) + len(sentence) <= self.settings.chunk_size:
                        current_chunk += sentence
                        chunk_sentences.append(sentence)
                    else:
                        # 保存当前chunk
                        if current_chunk:
                            chunk_content = current_chunk.strip()
                            # 始终保留文档标题作为主标题
                            if title:
                                if section_title:
                                    chunk_content = f"# {title}\n\n## {section_title}\n\n{chunk_content}"
                                else:
                                    chunk_content = f"# {title}\n\n{chunk_content}"
                            else:
                                chunk_content = f"## {section_title}\n\n{chunk_content}" if section_title else chunk_content
                            
                            chunks.append({
                                "content": chunk_content,
                                "section_title": section_title or "",
                                "chunk_index": len(chunks),
                                "original_para": ' '.join(chunk_sentences)
                            })
                        
                        # 开始新chunk
                        current_chunk = sentence
                        chunk_sentences = [sentence]
                
                # 保存最后一个chunk
                if current_chunk:
                    chunk_content = current_chunk.strip()
                    # 始终保留文档标题作为主标题
                    if title:
                        if section_title:
                            chunk_content = f"# {title}\n\n## {section_title}\n\n{chunk_content}"
                        else:
                            chunk_content = f"# {title}\n\n{chunk_content}"
                    else:
                        chunk_content = f"## {section_title}\n\n{chunk_content}" if section_title else chunk_content
                    
                    chunks.append({
                        "content": chunk_content,
                        "section_title": section_title or "",
                        "chunk_index": len(chunks),
                        "original_para": ' '.join(chunk_sentences)
                    })
            
            current_pos = para_start + len(para)
        
        # 添加重叠（如果启用）
        if self.settings.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # 从前一个chunk的末尾取overlap长度的文本
                    prev_chunk = chunks[i - 1]
                    prev_content = prev_chunk["original_para"]
                    
                    if len(prev_content) > self.settings.chunk_overlap:
                        overlap_text = prev_content[-self.settings.chunk_overlap:].strip()
                        # 尝试在单词边界截取
                        if not overlap_text[0].isspace():
                            first_space = overlap_text.find(' ')
                            if first_space > 0:
                                overlap_text = overlap_text[first_space:].strip()
                        
                        # 合并重叠文本，保留标题
                        current_section_title = chunk.get("section_title", "")
                        if title:
                            if current_section_title:
                                new_content = f"# {title}\n\n## {current_section_title}\n\n{overlap_text}\n\n{chunk['original_para']}"
                            else:
                                new_content = f"# {title}\n\n{overlap_text}\n\n{chunk['original_para']}"
                        else:
                            if current_section_title:
                                new_content = f"## {current_section_title}\n\n{overlap_text}\n\n{chunk['original_para']}"
                            else:
                                new_content = f"{overlap_text}\n\n{chunk['original_para']}"
                        
                        overlapped_chunks.append({
                            "content": new_content,
                            "section_title": current_section_title,
                            "chunk_index": chunk["chunk_index"],
                            "original_para": chunk["original_para"]
                        })
                    else:
                        overlapped_chunks.append(chunk)
            
            return overlapped_chunks
        
        return chunks
    
    def load_knowledge_base(self, file_path: Optional[Path] = None, batch_size: int = 100):
        """
        从JSON文件加载知识库到向量数据库
        
        Args:
            file_path: 知识库JSON文件路径，默认使用配置路径
            batch_size: 批处理大小
        """
        file_path = file_path or self.settings.knowledge_base_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {file_path}")
        
        print(f"正在加载知识库: {file_path}")
        print(f"文件大小: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 加载JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"✓ 加载了 {len(documents)} 个文档")
        
        # 处理文档
        print("正在分析文档并分块...")
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        processed_count = 0
        skipped_count = 0
        
        for i, doc in enumerate(documents):
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(documents)} 个文档...")
            
            # 获取文档信息
            doc_id = doc.get("doc_id", f"doc_{i}")
            title = doc.get("title") or ""
            methods = doc.get("methods") or ""
            
            # 如果methods为空，跳过
            if not methods or not methods.strip():
                skipped_count += 1
                continue
            
            # 构建文档文本：title + methods
            # 注意：methods部分已经包含标题，所以直接使用methods即可
            # 但我们需要在分块时保留methods中的标题
            
            # 分块
            chunks = self._chunk_by_paragraphs(methods, title)
            
            if not chunks:
                skipped_count += 1
                continue
            
            # 为每个chunk创建metadata和ID
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                all_chunks.append(chunk["content"])
                all_ids.append(chunk_id)
                
                # 构建metadata
                def safe_str(value, max_len=None):
                    """安全地将值转换为字符串，处理None值"""
                    if value is None:
                        return ""
                    s = str(value)
                    if max_len and len(s) > max_len:
                        return s[:max_len]
                    return s
                
                all_metadatas.append({
                    "doc_id": safe_str(doc_id, 200),
                    "source": safe_str(doc.get("source"), 100),
                    "source_id": safe_str(doc.get("source_id"), 200),
                    "title": safe_str(title, 200),
                    "section_title": safe_str(chunk["section_title"], 200),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "has_methods": doc.get("metadata", {}).get("has_methods", False)
                })
            
            processed_count += 1
        
        print(f"\n处理完成: {processed_count} 个文档已处理，{skipped_count} 个文档被跳过")
        print(f"总共生成 {len(all_chunks)} 个chunks")
        
        # 批量添加到向量数据库
        if not all_chunks:
            print("⚠️ 没有可添加的chunks")
            return
        
        print("\n正在将chunks添加到向量数据库...")
        collection = self.collection
        batch_count = 0
        total_chunks = 0
        
        for batch_start in range(0, len(all_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]
            batch_ids = all_ids[batch_start:batch_end]
            batch_metadatas = all_metadatas[batch_start:batch_end]
            
            # 生成embeddings
            batch_embeddings = self.embedding_model.encode(batch_chunks).tolist()
            
            # 添加到集合
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_chunks,
                metadatas=batch_metadatas
            )
            
            batch_count += len(batch_chunks)
            total_chunks += len(batch_chunks)
            
            if batch_count % 500 == 0:
                print(f"    已添加 {batch_count}/{len(all_chunks)} 个chunks...")
        
        print(f"\n✓ 知识库加载完成！总共添加了 {total_chunks} 个chunks")
    
    def _rerank_results(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """
        使用Cross-Encoder重排检索结果
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            top_k: 返回前k个结果
        
        Returns:
            重排后的结果列表
        """
        if not candidates:
            return []
        
        # 如果cross-encoder可用，使用它进行重排
        if self.cross_encoder is not None:
            try:
                # 准备查询-文档对
                pairs = [[query, result.content] for result in candidates]
                
                # 使用cross-encoder计算相关性分数
                scores = self.cross_encoder.predict(pairs)
                
                # 创建重排结果
                reranked = []
                import numpy as np
                
                # 归一化所有分数到0-1范围（使用sigmoid）
                scores_array = np.array(scores)
                normalized_scores = 1 / (1 + np.exp(-scores_array))  # sigmoid归一化
                
                for i, result in enumerate(candidates):
                    cross_score = float(normalized_scores[i])
                    
                    # 结合原始相似度分数和cross-encoder分数
                    # 原始相似度 30% + cross-encoder 70%
                    final_score = result.score * 0.3 + cross_score * 0.7
                    
                    reranked.append(RetrievalResult(
                        document_id=result.document_id,
                        content=result.content,
                        metadata=result.metadata,
                        score=final_score,
                        source=result.source
                    ))
                
                # 按分数排序
                reranked.sort(key=lambda x: x.score, reverse=True)
                
                return reranked[:top_k]
            
            except Exception as e:
                print(f"⚠️ Cross-encoder重排失败: {e}，回退到简单重排")
                return self._simple_rerank(query, candidates, top_k)
        else:
            return self._simple_rerank(query, candidates, top_k)
    
    def _simple_rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """
        简单重排方法（回退方案）
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            top_k: 返回前k个结果
        
        Returns:
            重排后的结果列表
        """
        if not candidates:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # 计算重排分数
        reranked = []
        for result in candidates:
            content_lower = result.content.lower()
            
            # 1. 关键词匹配分数（0-1）
            matched_words = sum(1 for word in query_words if word in content_lower)
            keyword_score = matched_words / len(query_words) if query_words else 0
            
            # 2. 关键词位置权重（出现在标题中的权重更高）
            title_score = 0.0
            section_title = result.metadata.get("section_title", "").lower()
            for word in query_words:
                if word in section_title:
                    title_score += 1.0 / len(query_words)
            
            # 3. 原始相似度分数
            original_score = result.score
            
            # 4. 综合分数（加权平均）
            final_score = (
                original_score * 0.4 +
                keyword_score * 0.3 +
                title_score * 0.3
            )
            
            reranked.append(RetrievalResult(
                document_id=result.document_id,
                content=result.content,
                metadata=result.metadata,
                score=final_score,
                source=result.source
            ))
        
        # 按分数排序
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:top_k]
    
    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> RAGQueryResult:
        """
        执行检索 + 重排
        
        Args:
            query: 查询文本
            top_k: 返回的结果数
            similarity_threshold: 相似度阈值
        
        Returns:
            检索结果
        """
        start_time = time.time()
        
        # 使用配置默认值
        top_k = top_k or self.settings.rag_top_k
        similarity_threshold = similarity_threshold or self.settings.rag_similarity_threshold
        
        # 如果启用重排，先检索更多候选
        initial_top_k = self.settings.rerank_top_k if self.settings.enable_reranking else top_k
        final_top_k = self.settings.final_top_k if self.settings.enable_reranking else top_k
        
        # 生成query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        if not self.collection:
            return RAGQueryResult(
                query=query,
                results=[],
                retrieval_time_ms=0.0
            )
        
        results = []
        
        # 执行检索
        try:
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_top_k
            )
            
            # 解析结果
            documents = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]
            distances = search_results.get("distances", [[]])[0]
            ids = search_results.get("ids", [[]])[0]
            
            # 转换为RetrievalResult（候选列表）
            candidates = []
            for i in range(len(documents)):
                # 将距离转换为相似度分数 (0-1)
                similarity_score = 1.0 / (1.0 + distances[i])
                
                # 应用相似度阈值
                if similarity_score < similarity_threshold:
                    continue
                
                result = RetrievalResult(
                    document_id=ids[i],
                    content=documents[i],
                    metadata=metadatas[i],
                    score=similarity_score,
                    source="core_papers_methods"
                )
                
                candidates.append(result)
            
            # 重排（如果启用）
            if self.settings.enable_reranking and len(candidates) > final_top_k:
                results = self._rerank_results(query, candidates, final_top_k)
            else:
                # 不使用重排，直接返回前top_k个
                results = candidates[:top_k]
        
        except Exception as e:
            print(f"⚠️  检索失败: {e}")
        
        # 计算检索时间
        retrieval_time = (time.time() - start_time) * 1000
        
        return RAGQueryResult(
            query=query,
            results=results,
            retrieval_time_ms=retrieval_time
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.collection:
            return {"error": "集合未初始化"}
        
        count = self.collection.count()
        
        return {
            "name": "core_papers_methods",
            "document_count": count,
            "metadata": self.collection.metadata
        }


# ==================== 导出 ====================

__all__ = [
    'CoreRAGSystem',
    'CoreRAGSettings',
    'RetrievalResult',
    'RAGQueryResult'
]


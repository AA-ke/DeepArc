"""
RAG/rag.py
混合RAG检索系统 - 结合共享和专业化知识库
从 Knowledge_Corpus/data/cleaned/documents_deduped.json 加载知识库
"""

import time
import json
from enum import Enum
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

class KnowledgeScope(Enum):
    """知识库作用域"""
    SHARED = "shared"  # 共享知识库
    SPECIALIZED = "specialized"  # 专业知识库


@dataclass
class KnowledgeSource:
    """知识源配置"""
    name: str
    collection_name: str
    description: str
    scope: KnowledgeScope
    agents: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 1


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
    shared_results: List[RetrievalResult] = field(default_factory=list)
    specialized_results: List[RetrievalResult] = field(default_factory=list)
    retrieval_time_ms: float = 0.0


# ==================== 配置类 ====================

class RAGSettings:
    """RAG系统配置"""
    def __init__(self):
        # Embedding模型
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # 向量数据库配置
        self.vector_db_type = "chromadb"
        self.chroma_persist_directory = "./RAG/chroma_db"
        
        # 检索配置
        self.rag_top_k = 5  # 减少到5，降低token使用
        self.rag_similarity_threshold = 0.3
        self.rag_default_strategy = "hybrid"
        
        # 文档分块配置
        self.chunk_size = 1024  # 每个chunk的字符数
        self.chunk_overlap = 100  # chunk之间的重叠字符数（增加重叠以保留更多上下文）
        
        # 重排配置
        self.enable_reranking = True  # 是否启用重排
        self.rerank_top_k = 10  # 重排候选数量（从top_k中选出，减少到10）
        self.final_top_k = 5  # 最终返回的结果数
        self.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder模型
        
        # 知识库路径
        self.knowledge_base_path = Path(__file__).parent.parent / "Knowledge_Corpus" / "data" / "cleaned" / "documents_deduped.json"
        self.core_papers_path = Path(__file__).parent.parent / "Knowledge_Corpus" / "data" / "raw" / "core_papers_md_methods.json"


# ==================== 混合RAG系统 ====================

class HybridRAGSystem:
    """
    混合RAG系统
    
    三层架构:
    - 第一层: 共享基础知识库（所有智能体）
    - 第二层: 专业知识库（特定智能体）
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化混合RAG系统
        
        Args:
            config: 可选的配置字典，覆盖默认配置
        """
        self.settings = RAGSettings()
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
        
        # 加载知识源配置
        self.knowledge_sources = self._get_default_knowledge_sources()
        
        # 初始化集合
        self.collections = {}
        self._initialize_collections()
    
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
    
    def _get_default_knowledge_sources(self) -> List[KnowledgeSource]:
        """获取默认知识源配置（单一共享向量库）"""
        return [
            # 共享知识库 - 所有agent共享使用
            KnowledgeSource(
                name="shared_knowledge_base",
                scope=KnowledgeScope.SHARED,
                collection_name="shared_knowledge_base",
                description="基因调控元件设计知识库（所有智能体共享）",
                agents=["all"],
                priority=1
            ),
        ]
    
    def _initialize_collections(self):
        """初始化所有向量数据库集合"""
        print("初始化向量数据库集合...")
        for source in self.knowledge_sources:
            if not source.enabled:
                continue
            
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name=source.collection_name,
                    metadata={
                        "scope": source.scope.value,
                        "description": source.description,
                        "agents": ",".join(source.agents),
                        "priority": str(source.priority)
                    }
                )
                self.collections[source.collection_name] = collection
                print(f"  ✓ {source.collection_name} ({source.scope.value})")
            
            except Exception as e:
                print(f"  ✗ {source.collection_name}: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        优化的文本分块策略
        
        策略：
        1. 优先按段落分割（双换行符）
        2. 其次按句子分割（句号、问号、感叹号）
        3. 最后按单换行符分割
        4. 确保chunk不超过指定大小
        5. 保留重叠以维持上下文
        
        Args:
            text: 要分块的文本
            chunk_size: 每个chunk的字符数
            overlap: chunk之间的重叠字符数
        
        Returns:
            文本块列表
        """
        chunk_size = chunk_size or self.settings.chunk_size
        overlap = overlap or self.settings.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        # 1. 按段落分割（双换行符或空行）
        paragraphs = []
        for para in text.split('\n\n'):
            para = para.strip()
            if para:
                paragraphs.append(para)
        
        # 如果只有一个段落或段落太大，按句子分割
        chunks = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                # 按句子分割（句号、问号、感叹号后跟空格或换行）
                import re
                sentences = re.split(r'([.!?]\s+)', para)
                # 重新组合句子（保留分隔符）
                current_sentence = ""
                for i in range(0, len(sentences), 2):
                    if i + 1 < len(sentences):
                        sentence = sentences[i] + sentences[i + 1]
                    else:
                        sentence = sentences[i]
                    
                    # 如果当前句子加上新句子不超过chunk_size，合并
                    if len(current_sentence) + len(sentence) <= chunk_size:
                        current_sentence += sentence
                    else:
                        if current_sentence:
                            chunks.append(current_sentence.strip())
                        # 如果单个句子就超过chunk_size，强制分割
                        if len(sentence) > chunk_size:
                            # 按单换行符或空格分割
                            words = sentence.split()
                            current_chunk = ""
                            for word in words:
                                if len(current_chunk) + len(word) + 1 <= chunk_size:
                                    current_chunk += " " + word if current_chunk else word
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                    current_chunk = word
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_sentence = ""
                        else:
                            current_sentence = sentence
                
                if current_sentence:
                    chunks.append(current_sentence.strip())
        
        # 2. 如果chunks仍然太大，进一步分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # 强制分割，但尽量在单词边界
                words = chunk.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk += " " + word if current_chunk else word
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = word
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
        
        # 3. 添加重叠（如果启用）
        if overlap > 0 and len(final_chunks) > 1:
            overlapped_chunks = [final_chunks[0]]
            for i in range(1, len(final_chunks)):
                prev_chunk = final_chunks[i - 1]
                current_chunk = final_chunks[i]
                
                # 从前一个chunk的末尾取overlap长度的文本
                if len(prev_chunk) > overlap:
                    overlap_text = prev_chunk[-overlap:]
                    # 尝试在单词边界截取
                    if not overlap_text[0].isspace():
                        first_space = overlap_text.find(' ')
                        if first_space > 0:
                            overlap_text = overlap_text[first_space:].strip()
                    
                    overlapped_chunk = overlap_text + " " + current_chunk
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(current_chunk)
            
            return overlapped_chunks
        
        return final_chunks
    
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
        import re
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
        按段落分块，保留标题（用于核心论文Methods部分）
        
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
        
        # 使用较大的chunk_size用于Methods部分（2048字符）
        methods_chunk_size = 2048
        
        for para_idx, para in enumerate(paragraphs):
            # 查找该段落对应的section标题
            para_start = full_text.find(para, current_pos)
            section_title = self._extract_section_title(full_text, para_start)
            
            # 如果段落太长，进一步分割
            if len(para) <= methods_chunk_size:
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
                import re
                sentences = re.split(r'([.!?]\s+)', para)
                current_chunk = ""
                chunk_sentences = []
                
                for i in range(0, len(sentences), 2):
                    if i + 1 < len(sentences):
                        sentence = sentences[i] + sentences[i + 1]
                    else:
                        sentence = sentences[i]
                    
                    # 如果当前chunk加上新句子不超过限制，合并
                    if len(current_chunk) + len(sentence) <= methods_chunk_size:
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
        
        # 添加重叠（如果启用，使用200字符重叠）
        methods_overlap = 200
        if methods_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # 从前一个chunk的末尾取overlap长度的文本
                    prev_chunk = chunks[i - 1]
                    prev_content = prev_chunk["original_para"]
                    
                    if len(prev_content) > methods_overlap:
                        overlap_text = prev_content[-methods_overlap:].strip()
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
    
    def _determine_collections(self, doc: Dict[str, Any]) -> List[str]:
        """
        确定文档应该放入哪个集合（单一共享库）
        
        Args:
            doc: 文档字典，包含 title, abstract, source 等字段
        
        Returns:
            集合名称列表（只有一个共享库）
        """
        # 所有文档都放入共享库
        return ["shared_knowledge_base"]
    
    def load_knowledge_base(self, file_path: Optional[Path] = None, batch_size: int = 100, load_core_papers: bool = True):
        """
        从JSON文件加载知识库到向量数据库
        
        Args:
            file_path: 知识库JSON文件路径，默认使用配置路径
            batch_size: 批处理大小
            load_core_papers: 是否同时加载核心论文Methods部分（默认True）
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
        
        # 按集合组织文档
        collection_docs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.collections.keys()}
        
        print("正在分析文档并分配到集合...")
        for i, doc in enumerate(documents):
            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i + 1}/{len(documents)} 个文档...")
            
            # 确定应该放入哪些集合
            target_collections = self._determine_collections(doc)
            
            # 构建文档文本（title + abstract）
            # 处理None值的情况
            title = doc.get("title") or ""
            abstract = doc.get("abstract") or ""
            title = str(title).strip() if title else ""
            abstract = str(abstract).strip() if abstract else ""
            full_text = f"{title}\n\n{abstract}".strip()
            
            if not full_text:
                continue
            
            # 为每个目标集合添加文档
            for coll_name in target_collections:
                if coll_name in collection_docs:
                    collection_docs[coll_name].append({
                        "doc": doc,
                        "text": full_text
                    })
        
        # 将文档添加到向量数据库
        print("\n正在将文档添加到向量数据库...")
        total_chunks = 0
        
        for coll_name, doc_list in collection_docs.items():
            if not doc_list:
                print(f"  ⚠️ {coll_name}: 无文档")
                continue
            
            print(f"\n处理集合: {coll_name} ({len(doc_list)} 个文档)")
            
            # 分块并准备数据
            all_chunks = []
            all_metadatas = []
            all_ids = []
            
            for item in doc_list:
                doc = item["doc"]
                text = item["text"]
                
                # 分块
                chunks = self._chunk_text(text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc.get('doc_id', 'unknown')}_chunk_{chunk_idx}"
                    all_chunks.append(chunk)
                    all_ids.append(chunk_id)
                    
                    # 构建metadata，确保所有值都不是None（ChromaDB不接受None值）
                    def safe_str(value, max_len=None):
                        """安全地将值转换为字符串，处理None值"""
                        if value is None:
                            return ""
                        s = str(value)
                        if max_len and len(s) > max_len:
                            return s[:max_len]
                        return s
                    
                    all_metadatas.append({
                        "doc_id": safe_str(doc.get("doc_id"), 200),
                        "source": safe_str(doc.get("source"), 100),
                        "source_id": safe_str(doc.get("source_id"), 200),
                        "title": safe_str(doc.get("title"), 200),
                        "authors": safe_str(doc.get("authors"), 200),
                        "journal": safe_str(doc.get("journal"), 200),
                        "date": safe_str(doc.get("date"), 50),
                        "doi": safe_str(doc.get("doi"), 200),
                        "url": safe_str(doc.get("url"), 500),
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "doc_type": "general"  # 标记为通用文档
                    })
            
            # 批量添加（分批处理以避免内存问题）
            collection = self.collections[coll_name]
            batch_count = 0
            
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
            
            print(f"  ✓ {coll_name}: 添加了 {len(all_chunks)} 个chunks")
        
        print(f"\n✓ 通用知识库加载完成！总共添加了 {total_chunks} 个chunks")
        
        # 加载核心论文Methods部分
        if load_core_papers and self.settings.core_papers_path.exists():
            print("\n" + "="*60)
            print("正在加载核心论文Methods部分...")
            print("="*60)
            self.load_core_papers(batch_size=batch_size)
    
    def load_core_papers(self, file_path: Optional[Path] = None, batch_size: int = 100):
        """
        加载核心论文Methods部分到向量数据库
        
        Args:
            file_path: 核心论文JSON文件路径，默认使用配置路径
            batch_size: 批处理大小
        """
        file_path = file_path or self.settings.core_papers_path
        
        if not file_path.exists():
            print(f"⚠️ 核心论文文件不存在: {file_path}，跳过加载")
            return
        
        print(f"正在加载核心论文: {file_path}")
        print(f"文件大小: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 加载JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"✓ 加载了 {len(documents)} 个核心论文文档")
        
        # 获取共享库集合
        collection_name = "shared_knowledge_base"
        collection = self.collections.get(collection_name)
        
        if not collection:
            print(f"⚠️ 集合 {collection_name} 不存在，跳过加载")
            return
        
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
            doc_id = doc.get("doc_id", f"core_doc_{i}")
            title = doc.get("title") or ""
            methods = doc.get("methods") or ""
            
            # 如果methods为空，跳过
            if not methods or not methods.strip():
                skipped_count += 1
                continue
            
            # 使用特殊的分块策略（按段落分块，保留标题）
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
                    "doc_type": "core_paper_methods",  # 标记为核心论文Methods
                    "has_methods": doc.get("metadata", {}).get("has_methods", False) if isinstance(doc.get("metadata"), dict) else False
                })
            
            processed_count += 1
        
        print(f"\n处理完成: {processed_count} 个文档已处理，{skipped_count} 个文档被跳过")
        print(f"总共生成 {len(all_chunks)} 个chunks")
        
        # 批量添加到向量数据库
        if not all_chunks:
            print("⚠️ 没有可添加的chunks")
            return
        
        print(f"\n正在将chunks添加到向量数据库集合: {collection_name}...")
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
        
        print(f"\n✓ 核心论文Methods部分加载完成！总共添加了 {total_chunks} 个chunks")
    
    def get_agent_collections(self, agent_role: str) -> List[str]:
        """
        获取智能体可访问的知识库集合（所有agent共享同一个库）
        
        Args:
            agent_role: 智能体角色名称（此参数保留以兼容接口，但所有agent都访问同一个库）
        
        Returns:
            集合名称列表（只有一个共享库）
        """
        # 所有agent都访问同一个共享库
        return ["shared_knowledge_base"]
    
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
                # 回退到简单重排
                return self._simple_rerank(query, candidates, top_k)
        else:
            # 使用简单重排
            return self._simple_rerank(query, candidates, top_k)
    
    def _simple_rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """
        简单重排方法（回退方案）
        
        使用关键词匹配和位置权重进行重排
        
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
            
            # 2. 关键词位置权重（出现在开头的权重更高）
            position_score = 0.0
            for word in query_words:
                pos = content_lower.find(word)
                if pos >= 0:
                    # 位置越靠前，分数越高
                    position_score += (1.0 - min(pos / len(content_lower), 0.5)) / len(query_words)
            
            # 3. 原始相似度分数
            original_score = result.score
            
            # 4. 综合分数（加权平均）
            # 原始相似度 40% + 关键词匹配 30% + 位置权重 30%
            final_score = (
                original_score * 0.4 +
                keyword_score * 0.3 +
                position_score * 0.3
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
        agent_role: str = None,
        top_k: int = None,
        use_shared: bool = True,
        use_specialized: bool = True,
        similarity_threshold: float = None
    ) -> RAGQueryResult:
        """
        执行检索（单一共享向量库）+ 重排
        
        Args:
            query: 查询文本
            agent_role: 智能体角色（保留以兼容接口，但不再使用）
            top_k: 返回的结果数
            use_shared: 是否使用共享知识库（保留以兼容接口，始终为True）
            use_specialized: 是否使用专业知识库（保留以兼容接口，不再使用）
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
        
        # 获取共享库集合
        collection_name = "shared_knowledge_base"
        collection = self.collections.get(collection_name)
        
        if not collection:
            return RAGQueryResult(
                query=query,
                shared_results=[],
                specialized_results=[],
                retrieval_time_ms=0.0
            )
        
        shared_results = []
        
        # 执行检索
        try:
            search_results = collection.query(
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
                    source="shared_knowledge_base"
                )
                
                candidates.append(result)
            
            # 重排（如果启用）
            if self.settings.enable_reranking and len(candidates) > final_top_k:
                shared_results = self._rerank_results(query, candidates, final_top_k)
            else:
                # 不使用重排，直接返回前top_k个
                shared_results = candidates[:top_k]
        
        except Exception as e:
            print(f"⚠️  检索集合 {collection_name} 失败: {e}")
        
        # 计算检索时间
        retrieval_time = (time.time() - start_time) * 1000
        
        return RAGQueryResult(
            query=query,
            shared_results=shared_results,
            specialized_results=[],  # 不再使用专业库
            retrieval_time_ms=retrieval_time
        )
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        if collection_name not in self.collections:
            return {"error": "集合不存在"}
        
        collection = self.collections[collection_name]
        count = collection.count()
        
        return {
            "name": collection_name,
            "document_count": count,
            "metadata": collection.metadata
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有集合的统计信息"""
        stats = {
            "total_collections": len(self.collections),
            "total_documents": 0,
            "collections": []
        }
        
        for collection_name in self.collections:
            col_stats = self.get_collection_stats(collection_name)
            stats["collections"].append(col_stats)
            stats["total_documents"] += col_stats.get("document_count", 0)
        
        return stats


# ==================== 智能体RAG接口 ====================

class AgentRAGInterface:
    """智能体专用的RAG查询接口"""
    
    def __init__(self, rag_system: HybridRAGSystem, agent_role: str):
        """
        初始化智能体RAG接口
        
        Args:
            rag_system: 混合RAG系统实例
            agent_role: 智能体角色（data_management, methodology, model_architect, result_analyst）
        """
        self.rag_system = rag_system
        self.agent_role = agent_role
        self.settings = rag_system.settings
    
    async def query(
        self,
        query: str,
        strategy: str = None,
        top_k: int = None
    ) -> RAGQueryResult:
        """
        执行RAG查询（单一共享向量库）
        
        Args:
            query: 查询文本
            strategy: 检索策略（保留以兼容接口，但不再使用，所有查询都使用共享库）
            top_k: 返回结果数
        
        Returns:
            检索结果
        """
        # 直接使用共享库检索
        return await self.rag_system.retrieve(
            query=query,
            agent_role=self.agent_role,
            top_k=top_k,
            use_shared=True,
            use_specialized=False
        )


# ==================== 导出 ====================

__all__ = [
    'HybridRAGSystem',
    'AgentRAGInterface',
    'RAGSettings',
    'KnowledgeScope',
    'KnowledgeSource',
    'RetrievalResult',
    'RAGQueryResult'
]

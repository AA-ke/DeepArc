"""
RAG/benchmark.py
RAG系统评估脚本 - 计算答案质量和检索质量指标
"""

import json
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import re
from collections import Counter

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 导入RAG模块（使用相对导入）
try:
    from RAG.rag import HybridRAGSystem, RetrievalResult
except ImportError:
    # 如果相对导入失败，尝试直接导入
    import importlib.util
    rag_path = script_dir / "rag.py"
    spec = importlib.util.spec_from_file_location("rag", rag_path)
    rag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_module)
    HybridRAGSystem = rag_module.HybridRAGSystem
    RetrievalResult = rag_module.RetrievalResult


# ==================== 数据模型 ====================

@dataclass
class EvaluationResult:
    """单个QA pair的评估结果"""
    question: str
    ground_truth_answer: str
    no_rag_answer: str = ""
    rag_answer: str = ""
    retrieved_documents: List[RetrievalResult] = field(default_factory=list)
    # 答案正确性指标（多种定量指标）
    # 语义相似度
    rag_answer_correctness_semantic: float = 0.0  # 基于embedding的语义相似度
    no_rag_answer_correctness_semantic: float = 0.0
    # BLEU分数
    rag_answer_correctness_bleu: float = 0.0  # BLEU-4分数
    no_rag_answer_correctness_bleu: float = 0.0
    # ROUGE分数
    rag_answer_correctness_rouge_l: float = 0.0  # ROUGE-L分数
    no_rag_answer_correctness_rouge_l: float = 0.0
    # 编辑距离
    rag_answer_correctness_edit_distance: float = 0.0  # 归一化的编辑距离相似度
    no_rag_answer_correctness_edit_distance: float = 0.0
    # Jaccard相似度（词汇重叠）
    rag_answer_correctness_jaccard: float = 0.0  # Jaccard相似度
    no_rag_answer_correctness_jaccard: float = 0.0
    # 字符级别相似度
    rag_answer_correctness_char_sim: float = 0.0  # 字符级别相似度
    no_rag_answer_correctness_char_sim: float = 0.0
    # 其他指标
    rag_faithfulness: float = 0.0  # 答案对检索文档的忠实度
    # 检索质量指标
    retrieval_precision: float = 0.0  # 检索到的文档中包含ground truth文档的比例
    retrieval_recall: float = 0.0  # 检索到的文档中ground truth文档的排名
    evaluation_time_ms: float = 0.0


@dataclass
class BenchmarkStats:
    """整体评估统计"""
    total_questions: int = 0
    avg_retrieval_time_ms: float = 0.0
    avg_evaluation_time_ms: float = 0.0
    results: List[EvaluationResult] = field(default_factory=list)


# ==================== 配置 ====================

QA_PAIRS_FILE = Path(__file__).parent.parent / "Knowledge_Corpus" / "data" / "qa_pairs.json"
SOURCE_DOCUMENTS_FILE = Path(__file__).parent.parent / "Knowledge_Corpus" / "data" / "cleaned" / "documents_deduped.json"
OUTPUT_FILE = Path(__file__).parent / "benchmark_results.json"
PROGRESS_FILE = Path(__file__).parent / "benchmark_progress.json"

# 评估配置
MAX_EVALUATION_QUESTIONS = 500  # None 表示评估所有问题
BATCH_SIZE = 10  # 每批处理的问题数量
DELAY_BETWEEN_BATCHES = 1.0  # 批次间延迟（秒）


# ==================== 工具函数 ====================

def load_qa_pairs(file_path: Path) -> List[Dict[str, Any]]:
    """加载QA pairs"""
    print(f"📖 正在加载 QA pairs: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    print(f"✓ 已加载 {len(qa_pairs)} 个 QA pairs")
    return qa_pairs


def load_source_documents(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """加载原始文档，返回doc_id到文档的映射"""
    print(f"📖 正在加载原始文档: {file_path}")
    if not file_path.exists():
        print(f"⚠️ 原始文档文件不存在: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # 构建doc_id到文档的映射
    doc_map = {}
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        if doc_id:
            doc_map[doc_id] = doc
    
    print(f"✓ 已加载 {len(doc_map)} 个原始文档")
    return doc_map


def create_retrieval_result_from_doc(doc: Dict[str, Any], doc_id: str) -> RetrievalResult:
    """从原始文档创建RetrievalResult对象"""
    title = doc.get("title") or ""
    abstract = doc.get("abstract") or ""
    title = str(title).strip() if title else ""
    abstract = str(abstract).strip() if abstract else ""
    content = f"{title}\n\n{abstract}".strip()
    
    return RetrievalResult(
        document_id=doc_id,
        content=content,
        metadata={
            "doc_id": doc.get("doc_id", ""),
            "source": doc.get("source", ""),
            "source_id": doc.get("source_id", ""),
            "title": doc.get("title", ""),
            "authors": doc.get("authors", ""),
            "journal": doc.get("journal", ""),
            "date": doc.get("date", ""),
            "doi": doc.get("doi", ""),
            "url": doc.get("url", ""),
        },
        score=1.0,  # 原始文档的分数设为1.0
        source="source_document"
    )


def load_progress() -> Dict[str, Any]:
    """加载进度信息"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed_indices": [],
        "last_processed_index": -1,
        "start_time": datetime.now().isoformat()
    }


def save_progress(progress: Dict[str, Any]):
    """保存进度信息"""
    progress["last_update"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_results(results: List[EvaluationResult], output_file: Path):
    """保存评估结果"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化的格式
    serializable_results = []
    for result in results:
            serializable_results.append({
            "question": result.question,
            "ground_truth_answer": result.ground_truth_answer,
            "no_rag_answer": result.no_rag_answer,
            "rag_answer": result.rag_answer,
            # RAG指标
            "rag_retrieved_documents": [
                {
                    "document_id": doc.document_id,
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "score": doc.score,
                    "source": doc.source
                }
                for doc in result.retrieved_documents
            ],
            # RAG正确性指标
            "rag_answer_correctness_semantic": result.rag_answer_correctness_semantic,
            "rag_answer_correctness_bleu": result.rag_answer_correctness_bleu,
            "rag_answer_correctness_rouge_l": result.rag_answer_correctness_rouge_l,
            "rag_answer_correctness_edit_distance": result.rag_answer_correctness_edit_distance,
            "rag_answer_correctness_jaccard": result.rag_answer_correctness_jaccard,
            "rag_answer_correctness_char_sim": result.rag_answer_correctness_char_sim,
            "rag_faithfulness": result.rag_faithfulness,
            "retrieval_precision": result.retrieval_precision,
            "retrieval_recall": result.retrieval_recall,
            # 无RAG正确性指标
            "no_rag_answer_correctness_semantic": result.no_rag_answer_correctness_semantic,
            "no_rag_answer_correctness_bleu": result.no_rag_answer_correctness_bleu,
            "no_rag_answer_correctness_rouge_l": result.no_rag_answer_correctness_rouge_l,
            "no_rag_answer_correctness_edit_distance": result.no_rag_answer_correctness_edit_distance,
            "no_rag_answer_correctness_jaccard": result.no_rag_answer_correctness_jaccard,
            "no_rag_answer_correctness_char_sim": result.no_rag_answer_correctness_char_sim,
            "evaluation_time_ms": result.evaluation_time_ms
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)


# ==================== LLM 调用函数 ====================

async def generate_answer_without_rag(
    llm: ChatOpenAI,
    question: str
) -> str:
    """不使用RAG生成答案"""
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on your knowledge."),
        HumanMessage(content=f"Question: {question}\n\nPlease provide a clear and concise answer.")
    ]
    
    response = await llm.ainvoke(messages)
    return response.content.strip()


async def generate_answer_with_rag(
    llm: ChatOpenAI,
    question: str,
    retrieved_docs: List[RetrievalResult]
) -> str:
    """使用RAG生成答案"""
    # 构建上下文
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[Document {i}]\n{doc.content}")
    
    context = "\n\n".join(context_parts)
    
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided context documents. Use the information from the context to answer the question accurately."),
        HumanMessage(content=f"Context Documents:\n{context}\n\nQuestion: {question}\n\nPlease provide a clear and concise answer based on the context documents.")
    ]
    
    response = await llm.ainvoke(messages)
    return response.content.strip()


# ==================== 指标计算 ====================

def calculate_bleu_score(candidate: str, reference: str, n: int = 4) -> float:
    """
    计算BLEU分数（简化版，基于n-gram重叠）
    
    Args:
        candidate: 候选答案
        reference: 参考答案（ground truth）
        n: n-gram的最大n值（默认4，即BLEU-4）
    
    Returns:
        BLEU分数 (0-1)
    """
    def get_ngrams(text: str, n: int) -> List[Tuple]:
        """获取n-gram列表"""
        words = text.lower().split()
        if len(words) < n:
            return [tuple(words)]
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    if not candidate or not reference:
        return 0.0
    
    candidate_ngrams = get_ngrams(candidate, n)
    reference_ngrams = get_ngrams(reference, n)
    
    if not candidate_ngrams or not reference_ngrams:
        return 0.0
    
    # 计算精确匹配的n-gram数量
    candidate_counter = Counter(candidate_ngrams)
    reference_counter = Counter(reference_ngrams)
    
    matches = sum(min(candidate_counter[ngram], reference_counter[ngram]) 
                  for ngram in candidate_counter)
    
    # BLEU分数 = 匹配数 / 候选n-gram总数
    bleu = matches / len(candidate_ngrams) if candidate_ngrams else 0.0
    
    # 应用长度惩罚（简化版）
    if len(candidate.split()) < len(reference.split()):
        brevity_penalty = len(candidate.split()) / len(reference.split())
        bleu *= brevity_penalty
    
    return max(0.0, min(1.0, bleu))


def calculate_rouge_l(candidate: str, reference: str) -> float:
    """
    计算ROUGE-L分数（基于最长公共子序列LCS）
    
    Args:
        candidate: 候选答案
        reference: 参考答案（ground truth）
    
    Returns:
        ROUGE-L分数 (0-1)
    """
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列的长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    if not candidate or not reference:
        return 0.0
    
    candidate_words = candidate.lower().split()
    reference_words = reference.lower().split()
    
    if not candidate_words or not reference_words:
        return 0.0
    
    lcs = lcs_length(candidate_words, reference_words)
    
    # ROUGE-L = LCS长度 / 参考答案长度
    rouge_l = lcs / len(reference_words) if reference_words else 0.0
    
    return max(0.0, min(1.0, rouge_l))


def calculate_edit_distance_similarity(candidate: str, reference: str) -> float:
    """
    计算基于编辑距离的相似度（Levenshtein距离）
    
    Args:
        candidate: 候选答案
        reference: 参考答案（ground truth）
    
    Returns:
        归一化的相似度分数 (0-1)，1表示完全相同，0表示完全不同
    """
    def levenshtein_distance(s1: str, s2: str) -> int:
        """计算Levenshtein编辑距离"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    if not candidate and not reference:
        return 1.0
    
    if not candidate or not reference:
        return 0.0
    
    # 计算编辑距离
    edit_dist = levenshtein_distance(candidate.lower(), reference.lower())
    
    # 归一化：相似度 = 1 - (编辑距离 / 最大长度)
    max_len = max(len(candidate), len(reference))
    similarity = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
    
    return max(0.0, min(1.0, similarity))


def calculate_jaccard_similarity(candidate: str, reference: str) -> float:
    """
    计算Jaccard相似度（基于词汇集合的重叠）
    
    Args:
        candidate: 候选答案
        reference: 参考答案（ground truth）
    
    Returns:
        Jaccard相似度 (0-1)
    """
    if not candidate or not reference:
        return 0.0
    
    candidate_words = set(candidate.lower().split())
    reference_words = set(reference.lower().split())
    
    if not candidate_words and not reference_words:
        return 1.0
    
    if not candidate_words or not reference_words:
        return 0.0
    
    # Jaccard = |A ∩ B| / |A ∪ B|
    intersection = len(candidate_words & reference_words)
    union = len(candidate_words | reference_words)
    
    jaccard = intersection / union if union > 0 else 0.0
    
    return max(0.0, min(1.0, jaccard))


def calculate_char_similarity(candidate: str, reference: str) -> float:
    """
    计算字符级别的相似度（基于字符集合的Jaccard相似度）
    
    Args:
        candidate: 候选答案
        reference: 参考答案（ground truth）
    
    Returns:
        字符级别相似度 (0-1)
    """
    if not candidate or not reference:
        return 0.0
    
    candidate_chars = set(candidate.lower().replace(" ", ""))
    reference_chars = set(reference.lower().replace(" ", ""))
    
    if not candidate_chars and not reference_chars:
        return 1.0
    
    if not candidate_chars or not reference_chars:
        return 0.0
    
    # Jaccard相似度
    intersection = len(candidate_chars & reference_chars)
    union = len(candidate_chars | reference_chars)
    
    similarity = intersection / union if union > 0 else 0.0
    
    return max(0.0, min(1.0, similarity))


async def calculate_answer_correctness_semantic(
    embeddings_model,
    answer: str,
    ground_truth: str
) -> float:
    """
    计算答案与ground truth的语义相似度（基于embedding）
    使用embedding计算余弦相似度
    
    Returns:
        语义相似度分数 (0-1)
    """
    try:
        # 生成embedding
        answer_embedding = embeddings_model.embed_query(answer)
        gt_embedding = embeddings_model.embed_query(ground_truth)
        
        # 计算余弦相似度
        similarity = cosine_similarity(
            [answer_embedding],
            [gt_embedding]
        )[0][0]
        
        # 归一化到0-1范围（余弦相似度范围是-1到1）
        normalized_similarity = (similarity + 1) / 2
        
        return max(0.0, min(1.0, normalized_similarity))
    except Exception as e:
        print(f"    ⚠️ 计算语义相似度失败: {e}")
        return 0.0


async def calculate_faithfulness(
    llm: ChatOpenAI,
    question: str,
    answer: str,
    retrieved_documents: List[RetrievalResult]
) -> float:
    """
    计算答案对检索文档的忠实度（Faithfulness）
    评估答案是否完全基于检索到的文档，没有引入外部知识或幻觉
    
    Returns:
        忠实度分数 (0-1)
    """
    if not retrieved_documents:
        return 0.0
    
    # 构建文档上下文
    context_parts = []
    for i, doc in enumerate(retrieved_documents, 1):
        context_parts.append(f"[Document {i}]\n{doc.content}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""Evaluate whether the answer is fully supported by the retrieved context documents.
Rate the faithfulness on a scale of 0 to 1, where:
- 1.0: The answer is completely supported by the context, with no unsupported claims
- 0.5: The answer is partially supported, but contains some unsupported information
- 0.0: The answer contains significant unsupported claims or contradicts the context

Question: {question}

Retrieved Context Documents:
{context}

Answer: {answer}

Output ONLY a single float number between 0 and 1, without any explanation."""

    messages = [
        SystemMessage(content="You are an expert at evaluating answer faithfulness to source documents."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"    ⚠️ 计算忠实度失败: {e}")
        return 0.0


def calculate_retrieval_metrics(
    retrieved_documents: List[RetrievalResult],
    ground_truth_doc_id: str
) -> Tuple[float, float]:
    """
    计算检索质量指标
    
    Args:
        retrieved_documents: 检索到的文档列表
        ground_truth_doc_id: ground truth文档的doc_id
    
    Returns:
        (precision, recall)
        - precision: 检索到的文档中包含ground truth文档的比例（如果ground truth在top-k中）
        - recall: ground truth文档在检索结果中的排名倒数（1/rank，如果找到的话）
    """
    if not retrieved_documents:
        return 0.0, 0.0
    
    # 检查ground truth文档是否在检索结果中
    found_gt = False
    gt_rank = -1
    
    for i, doc in enumerate(retrieved_documents):
        # 检查doc_id是否匹配（可能在不同的chunk中）
        doc_id_from_metadata = doc.metadata.get("doc_id", "")
        if doc_id_from_metadata == ground_truth_doc_id:
            found_gt = True
            gt_rank = i + 1  # 排名从1开始
            break
    
    # Precision: 如果找到ground truth，precision = 1/检索到的文档数
    # 这表示"检索到的文档中有多少比例是ground truth"
    if found_gt:
        precision = 1.0 / len(retrieved_documents)
    else:
        precision = 0.0
    
    # Recall: 如果找到ground truth，recall = 1/rank（排名越靠前，recall越高）
    # 如果没找到，recall = 0
    if found_gt:
        recall = 1.0 / gt_rank
    else:
        recall = 0.0
    
    return precision, recall


# ==================== 评估主函数 ====================

async def evaluate_single_qa(
    llm: ChatOpenAI,
    rag_system: HybridRAGSystem,
    qa_pair: Dict[str, Any],
    qa_index: int,
    total_qa: int,
    source_docs_map: Dict[str, Dict[str, Any]] = None,
    embeddings_model = None
) -> EvaluationResult:
    """评估单个QA pair"""
    import time
    start_time = time.time()
    
    question = qa_pair["question"]
    ground_truth_answer = qa_pair["answer"]
    
    print(f"\n[{qa_index + 1}/{total_qa}] 评估问题: {question[:80]}...")
    
    result = EvaluationResult(
        question=question,
        ground_truth_answer=ground_truth_answer
    )
    
    # 获取doc_id用于检索质量评估
    doc_id = qa_pair.get("doc_id", "")
    
    try:
        # 1. 不使用RAG生成答案
        print("  📝 生成无RAG答案...")
        result.no_rag_answer = await generate_answer_without_rag(llm, question)
        
        # 2. 使用RAG检索文档
        print("  🔍 检索相关文档...")
        rag_result = await rag_system.retrieve(
            query=question,
            top_k=5,
            similarity_threshold=0.3
        )
        result.retrieved_documents = rag_result.shared_results
        print(f"  ✓ 检索到 {len(result.retrieved_documents)} 个文档")
        
        # 3. 使用RAG生成答案
        print("  📝 生成RAG答案...")
        result.rag_answer = await generate_answer_with_rag(llm, question, result.retrieved_documents)
        
        # 4. 计算答案正确性指标（多种定量指标）
        print("  📊 计算答案正确性指标...")
        
        # 语义相似度（基于embedding）
        if embeddings_model:
            result.rag_answer_correctness_semantic = await calculate_answer_correctness_semantic(
                embeddings_model, result.rag_answer, ground_truth_answer
            )
            result.no_rag_answer_correctness_semantic = await calculate_answer_correctness_semantic(
                embeddings_model, result.no_rag_answer, ground_truth_answer
            )
        
        # BLEU分数
        result.rag_answer_correctness_bleu = calculate_bleu_score(
            result.rag_answer, ground_truth_answer
        )
        result.no_rag_answer_correctness_bleu = calculate_bleu_score(
            result.no_rag_answer, ground_truth_answer
        )
        
        # ROUGE-L分数
        result.rag_answer_correctness_rouge_l = calculate_rouge_l(
            result.rag_answer, ground_truth_answer
        )
        result.no_rag_answer_correctness_rouge_l = calculate_rouge_l(
            result.no_rag_answer, ground_truth_answer
        )
        
        # 编辑距离相似度
        result.rag_answer_correctness_edit_distance = calculate_edit_distance_similarity(
            result.rag_answer, ground_truth_answer
        )
        result.no_rag_answer_correctness_edit_distance = calculate_edit_distance_similarity(
            result.no_rag_answer, ground_truth_answer
        )
        
        # Jaccard相似度
        result.rag_answer_correctness_jaccard = calculate_jaccard_similarity(
            result.rag_answer, ground_truth_answer
        )
        result.no_rag_answer_correctness_jaccard = calculate_jaccard_similarity(
            result.no_rag_answer, ground_truth_answer
        )
        
        # 字符级别相似度
        result.rag_answer_correctness_char_sim = calculate_char_similarity(
            result.rag_answer, ground_truth_answer
        )
        result.no_rag_answer_correctness_char_sim = calculate_char_similarity(
            result.no_rag_answer, ground_truth_answer
        )
        
        # 忠实度（仅RAG）
        if result.retrieved_documents:
            result.rag_faithfulness = await calculate_faithfulness(
                llm, question, result.rag_answer, result.retrieved_documents
            )
        
        # 检索质量指标
        if doc_id:
            result.retrieval_precision, result.retrieval_recall = calculate_retrieval_metrics(
                result.retrieved_documents, doc_id
            )
        
        # 输出指标
        if embeddings_model:
            print(f"    ✓ RAG Semantic Similarity: {result.rag_answer_correctness_semantic:.3f}")
            print(f"    ✓ 无RAG Semantic Similarity: {result.no_rag_answer_correctness_semantic:.3f}")
        print(f"    ✓ RAG BLEU-4: {result.rag_answer_correctness_bleu:.3f}")
        print(f"    ✓ 无RAG BLEU-4: {result.no_rag_answer_correctness_bleu:.3f}")
        print(f"    ✓ RAG ROUGE-L: {result.rag_answer_correctness_rouge_l:.3f}")
        print(f"    ✓ 无RAG ROUGE-L: {result.no_rag_answer_correctness_rouge_l:.3f}")
        print(f"    ✓ RAG Edit Distance Sim: {result.rag_answer_correctness_edit_distance:.3f}")
        print(f"    ✓ 无RAG Edit Distance Sim: {result.no_rag_answer_correctness_edit_distance:.3f}")
        print(f"    ✓ RAG Jaccard: {result.rag_answer_correctness_jaccard:.3f}")
        print(f"    ✓ 无RAG Jaccard: {result.no_rag_answer_correctness_jaccard:.3f}")
        print(f"    ✓ RAG Char Similarity: {result.rag_answer_correctness_char_sim:.3f}")
        print(f"    ✓ 无RAG Char Similarity: {result.no_rag_answer_correctness_char_sim:.3f}")
        if result.retrieved_documents:
            print(f"    ✓ RAG Faithfulness: {result.rag_faithfulness:.3f}")
        if doc_id:
            print(f"    ✓ Retrieval Precision: {result.retrieval_precision:.3f}")
            print(f"    ✓ Retrieval Recall: {result.retrieval_recall:.3f}")
        
    except Exception as e:
        print(f"  ❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    result.evaluation_time_ms = (time.time() - start_time) * 1000
    return result


async def main():
    """主函数"""
    print("=" * 80)
    print("RAG系统评估脚本")
    print("=" * 80)
    print()
    
    # 加载设置
    settings = get_settings()
    if not settings.openai_api_key:
        print("❌ 错误: 未找到 OPENAI_API_KEY，请在 .env 文件中设置")
        return
    
    # 初始化LLM
    llm_model = "gpt-4o-mini"
    print(f"🤖 初始化 LLM: {llm_model}")
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0.3,  # 降低温度以获得更稳定的评估结果
        openai_api_key=settings.openai_api_key
    )
    print("✓ LLM 初始化完成")
    
    # 初始化Embeddings模型（用于计算语义相似度）
    print(f"🔤 初始化 Embeddings 模型...")
    try:
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        print("✓ Embeddings 模型初始化完成")
    except Exception as e:
        print(f"⚠️ Embeddings 模型初始化失败: {e}，将跳过语义相似度计算")
        embeddings_model = None
    
    # 初始化RAG系统
    print("\n🔧 初始化 RAG 系统...")
    rag_system = HybridRAGSystem()
    
    # 检查向量数据库是否已加载
    stats = rag_system.get_all_stats()
    if stats["total_documents"] == 0:
        print("⚠️ 向量数据库为空，正在加载知识库...")
        rag_system.load_knowledge_base()
    else:
        print(f"✓ RAG 系统已就绪（{stats['total_documents']} 个文档）")
    
    print()
    
    # 加载QA pairs
    if not QA_PAIRS_FILE.exists():
        print(f"❌ 错误: QA pairs 文件不存在: {QA_PAIRS_FILE}")
        return
    
    qa_pairs = load_qa_pairs(QA_PAIRS_FILE)
    
    # 加载原始文档
    source_docs_map = load_source_documents(SOURCE_DOCUMENTS_FILE)
    
    # 限制评估数量
    if MAX_EVALUATION_QUESTIONS:
        qa_pairs = qa_pairs[:MAX_EVALUATION_QUESTIONS]
        print(f"📊 限制评估数量为: {MAX_EVALUATION_QUESTIONS}")
    
    # 加载进度
    progress = load_progress()
    start_index = progress.get("last_processed_index", -1) + 1
    
    if start_index >= len(qa_pairs):
        print("✓ 所有问题已评估完成！")
        return
    
    print(f"📊 进度信息:")
    print(f"  - 已评估: {start_index}/{len(qa_pairs)}")
    print(f"  - 待评估: {len(qa_pairs) - start_index}")
    print()
    
    # 分批评估
    all_results = []
    total_batches = (len(qa_pairs) - start_index + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        batch_start = start_index + batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(qa_pairs))
        
        print(f"\n📦 处理批次 {batch_num + 1}/{total_batches} (问题 {batch_start + 1}-{batch_end})")
        
        batch_results = []
        for i in range(batch_start, batch_end):
            qa_pair = qa_pairs[i]
            result = await evaluate_single_qa(
                llm, rag_system, qa_pair, i, len(qa_pairs), 
                source_docs_map, embeddings_model
            )
            batch_results.append(result)
            all_results.append(result)
            
            # 问题间短暂延迟
            if i < batch_end - 1:
                await asyncio.sleep(0.5)
        
        # 保存进度和结果
        progress["last_processed_index"] = batch_end - 1
        progress["processed_indices"] = list(range(batch_start, batch_end))
        save_progress(progress)
        
        # 增量保存结果
        save_results(all_results, OUTPUT_FILE)
        print(f"  ✓ 已保存批次结果（总计: {len(all_results)} 个结果）")
        
        # 批次间延迟
        if batch_num < total_batches - 1:
            print(f"  ⏳ 等待 {DELAY_BETWEEN_BATCHES} 秒...")
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)
    
    # 计算统计信息
    print("\n" + "=" * 80)
    print("📊 评估统计")
    print("=" * 80)
    
    valid_rag_results = [r for r in all_results if r.rag_answer]
    valid_no_rag_results = [r for r in all_results if r.no_rag_answer]
    
    # 只统计 precision > 0 的 RAG 结果（检索到 ground truth 文档的情况）
    precision_positive_rag_results = [r for r in valid_rag_results if r.retrieval_precision > 0]
    # 对于无RAG结果，使用相同的索引（对应相同的问题）
    precision_positive_no_rag_results = [r for r in valid_no_rag_results 
                                          if any(r.question == rag_r.question for rag_r in precision_positive_rag_results)]
    
    print(f"总评估问题数: {len(all_results)}")
    print(f"有效RAG评估数: {len(valid_rag_results)}")
    print(f"有效无RAG评估数: {len(valid_no_rag_results)}")
    print(f"检索成功数 (precision > 0): {len(precision_positive_rag_results)}")
    print()
    
    print("=" * 80)
    print("📊 统计结果（仅 precision > 0 的案例）")
    print("=" * 80)
    print()
    
    if precision_positive_rag_results:
        # 计算各种正确性指标的平均值（仅 precision > 0 的案例）
        rag_avg_semantic = sum(r.rag_answer_correctness_semantic for r in precision_positive_rag_results if r.rag_answer_correctness_semantic > 0) / max(1, sum(1 for r in precision_positive_rag_results if r.rag_answer_correctness_semantic > 0))
        rag_avg_bleu = sum(r.rag_answer_correctness_bleu for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_rouge_l = sum(r.rag_answer_correctness_rouge_l for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_edit_dist = sum(r.rag_answer_correctness_edit_distance for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_jaccard = sum(r.rag_answer_correctness_jaccard for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_char_sim = sum(r.rag_answer_correctness_char_sim for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_faithfulness = sum(r.rag_faithfulness for r in precision_positive_rag_results if r.rag_faithfulness > 0) / max(1, sum(1 for r in precision_positive_rag_results if r.rag_faithfulness > 0))
        rag_avg_retrieval_precision = sum(r.retrieval_precision for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_retrieval_recall = sum(r.retrieval_recall for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        
        print(f"📊 RAG指标 (基于 {len(precision_positive_rag_results)} 个检索成功案例):")
        print(f"  【答案正确性指标】")
        if rag_avg_semantic > 0:
            print(f"    - Semantic Similarity: {rag_avg_semantic:.3f}")
        print(f"    - BLEU-4: {rag_avg_bleu:.3f}")
        print(f"    - ROUGE-L: {rag_avg_rouge_l:.3f}")
        print(f"    - Edit Distance Similarity: {rag_avg_edit_dist:.3f}")
        print(f"    - Jaccard Similarity: {rag_avg_jaccard:.3f}")
        print(f"    - Character Similarity: {rag_avg_char_sim:.3f}")
        if rag_avg_faithfulness > 0:
            print(f"    - Faithfulness: {rag_avg_faithfulness:.3f}")
        print(f"  【检索质量指标】")
        print(f"    - Retrieval Precision: {rag_avg_retrieval_precision:.3f}")
        print(f"    - Retrieval Recall: {rag_avg_retrieval_recall:.3f}")
    else:
        print("⚠️ 没有检索成功的RAG评估结果 (precision > 0)")
    
    print()
    
    if precision_positive_no_rag_results:
        no_rag_avg_semantic = sum(r.no_rag_answer_correctness_semantic for r in precision_positive_no_rag_results if r.no_rag_answer_correctness_semantic > 0) / max(1, sum(1 for r in precision_positive_no_rag_results if r.no_rag_answer_correctness_semantic > 0))
        no_rag_avg_bleu = sum(r.no_rag_answer_correctness_bleu for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_rouge_l = sum(r.no_rag_answer_correctness_rouge_l for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_edit_dist = sum(r.no_rag_answer_correctness_edit_distance for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_jaccard = sum(r.no_rag_answer_correctness_jaccard for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_char_sim = sum(r.no_rag_answer_correctness_char_sim for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        
        print(f"📊 无RAG指标 (基于 {len(precision_positive_no_rag_results)} 个检索成功案例):")
        print(f"  【答案正确性指标】")
        if no_rag_avg_semantic > 0:
            print(f"    - Semantic Similarity: {no_rag_avg_semantic:.3f}")
        print(f"    - BLEU-4: {no_rag_avg_bleu:.3f}")
        print(f"    - ROUGE-L: {no_rag_avg_rouge_l:.3f}")
        print(f"    - Edit Distance Similarity: {no_rag_avg_edit_dist:.3f}")
        print(f"    - Jaccard Similarity: {no_rag_avg_jaccard:.3f}")
        print(f"    - Character Similarity: {no_rag_avg_char_sim:.3f}")
    else:
        print("⚠️ 没有检索成功的无RAG评估结果 (precision > 0)")
    
    print()
    
    # 对比分析（仅 precision > 0 的案例）
    if precision_positive_rag_results and precision_positive_no_rag_results:
        print("📈 对比分析 (RAG vs 无RAG, 仅检索成功案例):")
        rag_avg_semantic = sum(r.rag_answer_correctness_semantic for r in precision_positive_rag_results if r.rag_answer_correctness_semantic > 0) / max(1, sum(1 for r in precision_positive_rag_results if r.rag_answer_correctness_semantic > 0))
        rag_avg_bleu = sum(r.rag_answer_correctness_bleu for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_rouge_l = sum(r.rag_answer_correctness_rouge_l for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_edit_dist = sum(r.rag_answer_correctness_edit_distance for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_jaccard = sum(r.rag_answer_correctness_jaccard for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        rag_avg_char_sim = sum(r.rag_answer_correctness_char_sim for r in precision_positive_rag_results) / len(precision_positive_rag_results)
        
        no_rag_avg_semantic = sum(r.no_rag_answer_correctness_semantic for r in precision_positive_no_rag_results if r.no_rag_answer_correctness_semantic > 0) / max(1, sum(1 for r in precision_positive_no_rag_results if r.no_rag_answer_correctness_semantic > 0))
        no_rag_avg_bleu = sum(r.no_rag_answer_correctness_bleu for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_rouge_l = sum(r.no_rag_answer_correctness_rouge_l for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_edit_dist = sum(r.no_rag_answer_correctness_edit_distance for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_jaccard = sum(r.no_rag_answer_correctness_jaccard for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        no_rag_avg_char_sim = sum(r.no_rag_answer_correctness_char_sim for r in precision_positive_no_rag_results) / len(precision_positive_no_rag_results)
        
        print(f"  【答案正确性指标】")
        if rag_avg_semantic > 0 and no_rag_avg_semantic > 0:
            print(f"    - Semantic Similarity: RAG {rag_avg_semantic:.3f} vs 无RAG {no_rag_avg_semantic:.3f} (差异: {rag_avg_semantic - no_rag_avg_semantic:+.3f})")
        print(f"    - BLEU-4: RAG {rag_avg_bleu:.3f} vs 无RAG {no_rag_avg_bleu:.3f} (差异: {rag_avg_bleu - no_rag_avg_bleu:+.3f})")
        print(f"    - ROUGE-L: RAG {rag_avg_rouge_l:.3f} vs 无RAG {no_rag_avg_rouge_l:.3f} (差异: {rag_avg_rouge_l - no_rag_avg_rouge_l:+.3f})")
        print(f"    - Edit Distance Sim: RAG {rag_avg_edit_dist:.3f} vs 无RAG {no_rag_avg_edit_dist:.3f} (差异: {rag_avg_edit_dist - no_rag_avg_edit_dist:+.3f})")
        print(f"    - Jaccard: RAG {rag_avg_jaccard:.3f} vs 无RAG {no_rag_avg_jaccard:.3f} (差异: {rag_avg_jaccard - no_rag_avg_jaccard:+.3f})")
        print(f"    - Char Similarity: RAG {rag_avg_char_sim:.3f} vs 无RAG {no_rag_avg_char_sim:.3f} (差异: {rag_avg_char_sim - no_rag_avg_char_sim:+.3f})")
    
    if all_results:
        avg_eval_time = sum(r.evaluation_time_ms for r in all_results) / len(all_results)
        print(f"\n⏱️  平均评估时间: {avg_eval_time:.1f} ms")
    
    print(f"\n✓ 评估完成！结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())


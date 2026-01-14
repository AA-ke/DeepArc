import json
import os
import sys
import string
from datetime import datetime
from collections import defaultdict
import difflib

def load_unified_data():
    """加载统一格式的数据"""
    try:
        with open("Knowledge_Corpus/data/unified/all_documents_raw.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print("⚠️ 数据格式错误：期望列表", file=sys.stderr, flush=True)
                return []
            return data
    except FileNotFoundError:
        print("❌ 文件不存在: Knowledge_Corpus/data/unified/all_documents_raw.json", file=sys.stderr, flush=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


def calculate_title_similarity(title1, title2):
    """计算标题相似度"""
    if not title1 or not title2:
        return 0.0
    
    title1 = title1.lower().strip()
    title2 = title2.lower().strip()
    
    return difflib.SequenceMatcher(None, title1, title2).ratio()


def deduplicate_by_id(docs):
    """基于ID去重"""
    
    print("1️⃣ 基于ID去重...", flush=True)
    
    seen_ids = {}  # key: source_id, value: doc
    duplicates = []
    docs_without_id = []
    
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        
        source = doc.get("source", "")
        source_id = doc.get("source_id", "")
        
        if not source_id:
            docs_without_id.append(doc)
            continue
        
        # 构建复合key
        key = f"{source}:{source_id}"
        
        if key in seen_ids:
            duplicates.append(doc)
        else:
            seen_ids[key] = doc
    
    # 合并有ID和没有ID的文档
    result = list(seen_ids.values()) + docs_without_id
    
    print(f"   发现 {len(duplicates)} 个ID重复项", flush=True)
    if docs_without_id:
        print(f"   保留 {len(docs_without_id)} 个无ID文档", flush=True)
    
    return result, duplicates


def deduplicate_by_doi(docs):
    """基于DOI去重"""
    
    print("2️⃣ 基于DOI去重...", flush=True)
    
    seen_dois = {}
    duplicates = []
    unique_docs = []
    
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        
        # 安全获取DOI，处理None值
        doi_raw = doc.get("doi") or ""
        doi = str(doi_raw).strip().lower() if doi_raw else ""
        
        if doi and doi in seen_dois:
            duplicates.append(doc)
        else:
            unique_docs.append(doc)
            if doi:
                seen_dois[doi] = doc
    
    print(f"   发现 {len(duplicates)} 个DOI重复项", flush=True)
    
    return unique_docs, duplicates


def deduplicate_by_title(docs, similarity_threshold=0.9):
    """基于标题相似度去重（优化版本：使用字典加速）"""
    
    print(f"3️⃣ 基于标题相似度去重 (阈值: {similarity_threshold})...", flush=True)
    
    unique_docs = []
    duplicates = []
    seen_titles_dict = {}  # key: normalized_title, value: doc
    
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            unique_docs.append(doc)
            continue
        
        title = doc.get("title", "").strip()
        
        if not title:
            unique_docs.append(doc)
            continue
        
        # 标准化标题用于快速匹配（去除标点符号和多余空格）
        title_lower = title.lower()
        title_normalized = ''.join(c for c in title_lower if c not in string.punctuation)
        title_normalized = ' '.join(title_normalized.split())  # 规范化空格
        
        is_duplicate = False
        
        # 检查标准化标题是否已存在
        if title_normalized in seen_titles_dict:
            # 精确匹配，直接判定为重复
            duplicates.append(doc)
            is_duplicate = True
        else:
            # 检查相似度（只与已保存的标题比较，减少计算量）
            for seen_title_norm, seen_doc in seen_titles_dict.items():
                similarity = difflib.SequenceMatcher(None, title_normalized, seen_title_norm).ratio()
                
                if similarity >= similarity_threshold:
                    duplicates.append(doc)
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_docs.append(doc)
            seen_titles_dict[title_normalized] = doc
        
        if (i + 1) % 1000 == 0:
            print(f"   处理进度: {i+1}/{len(docs)}", flush=True)
    
    print(f"   发现 {len(duplicates)} 个标题相似项", flush=True)
    
    return unique_docs, duplicates


def analyze_duplicates(original_count, dedup_stages):
    """分析去重效果"""
    
    print(f"\n{'='*60}", flush=True)
    print("去重分析报告", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    print(f"原始文档数: {original_count:,}", flush=True)
    
    current_count = original_count
    for stage_name, removed_count in dedup_stages:
        current_count -= removed_count
        print(f"{stage_name}: 移除 {removed_count:,} 条 (剩余: {current_count:,})", flush=True)
    
    total_removed = original_count - current_count
    dedup_rate = (total_removed / original_count * 100) if original_count > 0 else 0
    
    print(f"\n总计移除: {total_removed:,} 条", flush=True)
    print(f"去重率: {dedup_rate:.2f}%", flush=True)
    print(f"最终文档数: {current_count:,}", flush=True)


if __name__ == "__main__":
    
    print("="*60, flush=True)
    print("步骤2: 数据去重", flush=True)
    print("="*60 + "\n", flush=True)
    
    try:
        # 加载数据
        docs = load_unified_data()
        original_count = len(docs)
        
        if original_count == 0:
            print("⚠️ 未加载到任何文档", flush=True)
            sys.exit(0)
        
        print(f"加载文档数: {original_count:,}\n", flush=True)
        
        dedup_stages = []
        
        # 1. ID去重
        docs, id_dups = deduplicate_by_id(docs)
        dedup_stages.append(("ID去重", len(id_dups)))
        print(f"   剩余文档: {len(docs):,}\n", flush=True)
        
        # 2. DOI去重
        docs, doi_dups = deduplicate_by_doi(docs)
        dedup_stages.append(("DOI去重", len(doi_dups)))
        print(f"   剩余文档: {len(docs):,}\n", flush=True)
        
        # 3. 标题去重
        docs, title_dups = deduplicate_by_title(docs, similarity_threshold=0.9)
        dedup_stages.append(("标题去重", len(title_dups)))
        print(f"   剩余文档: {len(docs):,}\n", flush=True)
        
        # 分析
        analyze_duplicates(original_count, dedup_stages)
        
        # 保存
        os.makedirs("Knowledge_Corpus/data/cleaned", exist_ok=True)
        os.makedirs("Knowledge_Corpus/data/metadata", exist_ok=True)
        
        try:
            with open("Knowledge_Corpus/data/cleaned/documents_deduped.json", "w", encoding="utf-8") as f:
                json.dump(docs, f, indent=2, ensure_ascii=False)
            print(f"\n💾 已保存到: Knowledge_Corpus/data/cleaned/documents_deduped.json", flush=True)
        except Exception as e:
            print(f"\n❌ 保存去重文档时出错: {e}", file=sys.stderr, flush=True)
        
        # 保存重复项（用于审查）
        all_duplicates = id_dups + doi_dups + title_dups
        if all_duplicates:
            try:
                with open("Knowledge_Corpus/data/cleaned/duplicates.json", "w", encoding="utf-8") as f:
                    json.dump(all_duplicates, f, indent=2, ensure_ascii=False)
                print(f"💾 重复项已保存到: Knowledge_Corpus/data/cleaned/duplicates.json", flush=True)
            except Exception as e:
                print(f"⚠️ 保存重复项时出错: {e}", file=sys.stderr, flush=True)
        
        # 保存统计
        try:
            with open("Knowledge_Corpus/data/metadata/02_dedup_stats.json", "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "original_count": original_count,
                    "final_count": len(docs),
                    "removed_count": original_count - len(docs),
                    "dedup_rate": (original_count - len(docs)) / original_count * 100 if original_count > 0 else 0,
                    "stages": [{"name": name, "removed": count} for name, count in dedup_stages]
                }, f, indent=2, ensure_ascii=False)
            print(f"📊 统计已保存到: Knowledge_Corpus/data/metadata/02_dedup_stats.json", flush=True)
        except Exception as e:
            print(f"⚠️ 保存统计时出错: {e}", file=sys.stderr, flush=True)
        
        print(f"\n✅ 完成！", flush=True)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
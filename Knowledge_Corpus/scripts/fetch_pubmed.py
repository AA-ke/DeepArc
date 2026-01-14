from Bio import Entrez
from Bio import Medline

import time
import sys
import json
import re
from pathlib import Path

# 设置你的邮箱（NCBI要求）
Entrez.email = "ake0906ake@gmail.com"

def search_pubmed(query, max_results=50):
    """搜索PubMed并获取PMID列表"""
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"搜索错误: {e}", file=sys.stderr)
        return []

def fetch_abstracts(pmid_list):
    """批量获取摘要"""
    if not pmid_list:
        return ""
    try:
        ids = ",".join(pmid_list)
        handle = Entrez.efetch(
            db="pubmed",
            id=ids,
            rettype="abstract",
            retmode="text"
        )
        result = handle.read()
        handle.close()
        return result
    except Exception as e:
        print(f"获取摘要错误: {e}", file=sys.stderr)
        return ""


def fetch_pubmed_records(pmid_list):
    """批量获取PubMed结构化记录（用于后续统一清洗）"""
    if not pmid_list:
        return []
    try:
        ids = ",".join(pmid_list)
        handle = Entrez.efetch(
            db="pubmed",
            id=ids,
            rettype="medline",
            retmode="text"
        )
        records = list(Medline.parse(handle))
        handle.close()

        parsed = []
        for rec in records:
            pmid = rec.get("PMID", "").strip()
            if not pmid:
                continue

            title = rec.get("TI", "").strip()
            abstract = rec.get("AB", "").strip()
            authors_list = rec.get("AU", [])
            authors = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)
            journal = rec.get("JT", "").strip() or rec.get("TA", "").strip()
            date = rec.get("DP", "").strip()

            doi = ""
            # LID 字段中可能包含 DOI
            lid = rec.get("LID", "")
            if isinstance(lid, list):
                lid_candidates = lid
            else:
                lid_candidates = [lid] if lid else []
            for lid_val in lid_candidates:
                m = re.search(r'(10\.\S+)', str(lid_val))
                if m:
                    doi = m.group(1)
                    break
            # AID 字段也可能有 DOI
            if not doi:
                aids = rec.get("AID", [])
                if isinstance(aids, list):
                    for a in aids:
                        if "[doi]" in a:
                            doi = a.split()[0]
                            break
                elif isinstance(aids, str) and "[doi]" in aids:
                    doi = aids.split()[0]

            parsed.append({
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "date": date,
            })

        return parsed
    except Exception as e:
        print(f"获取结构化记录错误: {e}", file=sys.stderr)
        return []


# 统一数据输出目录：相对于 Knowledge_Corpus 根目录
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

queries = [
    # 基础生物学知识
    "cis-regulatory element AND definition AND classification",
    "promoter enhancer silencer insulator AND structure AND function",
    "transcription factor binding site AND motif AND PWM",
    "gene regulatory elements AND transcriptional regulation AND mechanism",
    "chromatin accessibility AND ATAC-seq AND regulatory elements",
    "epigenetic modification AND gene regulatory elements AND histone modification",
    
    # 计算方法与预测
    "computational prediction AND enhancer activity AND promoter strength",
    "machine learning AND regulatory element prediction AND sequence-to-function",
    "deep learning AND cis-regulatory element AND CNN transformer",
    "MPRA AND massively parallel reporter assay AND regulatory element design",
    "STARR-seq AND enhancer screening AND high-throughput",
    
    # 设计方法
    "synthetic enhancer design AND de novo AND regulatory element",
    "synthetic promoter design AND engineered AND gene expression control",
    "regulatory sequence optimization AND inverse design AND directed evolution",
    "regulatory grammar AND sequence rules AND combinatorial design",
    
    # 实验验证与应用
    "designed regulatory elements AND experimental validation AND reporter assay",
    "gene regulatory element design AND gene therapy AND clinical application",
    "cell-type-specific enhancer AND tissue-specific promoter AND design",
    "regulatory element design AND off-target effect AND specificity",
    
    # 高级主题
    "3D genome AND chromatin looping AND enhancer-promoter interaction",
    "gene regulatory network AND cis-regulatory network AND systems biology",
    "evolutionary conservation AND regulatory element AND comparative genomics",
    "single-cell AND regulatory element AND cell-type-specific activity"
]

print("开始获取PubMed数据...", flush=True)

# 聚合所有查询的结果，最终只保存一个 JSON / TXT 文件
all_records = []
all_abstracts_text = []

for i, query in enumerate(queries):
    print(f"正在获取 ({i+1}/{len(queries)}): {query}", flush=True)
    try:
        pmids = search_pubmed(query, max_results=50)
        if not pmids:
            print(f"  警告: 查询 '{query}' 未找到结果", flush=True)
            continue
        
        print(f"  找到 {len(pmids)} 篇论文", flush=True)

        # 1) 结构化 JSON，用于后续统一清洗（累积到 all_records）
        records = fetch_pubmed_records(pmids)
        if records:
            all_records.extend(records)
            print(f"  ✅ 本次解析出 {len(records)} 条结构化记录（累计 {len(all_records)} 条）", flush=True)
        else:
            print(f"  警告: 未能获取结构化记录", flush=True)

        # 2) 可选：累积原始文本摘要，便于人工检查
        abstracts = fetch_abstracts(pmids)
        if abstracts:
            all_abstracts_text.append(f"=== Query {i+1}: {query} ===\n{abstracts}\n")
        else:
            print(f"  提示: 未获取到原始摘要文本（不影响 JSON 清洗流程）", flush=True)
        
        time.sleep(1)  # 遵守NCBI速率限制
    except Exception as e:
        print(f"  错误处理查询 '{query}': {e}", file=sys.stderr, flush=True)
        continue

# 统一写出一个 JSON 文件
if all_records:
    json_path = RAW_DIR / "pubmed_all.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 所有结构化记录已保存到 {json_path}（共 {len(all_records)} 条）", flush=True)
else:
    print("\n⚠️ 未获取到任何结构化记录，未生成 pubmed_all.json", flush=True)

# 可选：统一写出一个 TXT 文件
if all_abstracts_text:
    txt_path = RAW_DIR / "pubmed_all.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_abstracts_text))
    print(f"（可选）原始摘要文本已保存到 {txt_path}", flush=True)

print(f"\n✅ 完成！检查 {RAW_DIR} 目录", flush=True)
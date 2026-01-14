import json
import os
import sys
import re
from datetime import datetime
from collections import defaultdict
import hashlib

def load_json_safe(filepath):
    """安全加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 如果是字典且包含papers/repos/pipelines键，提取列表
            if isinstance(data, dict):
                if "papers" in data:
                    return data["papers"]
                elif "repos" in data:
                    return data["repos"]
                elif "pipelines" in data:
                    return data["pipelines"]
                elif "collection" in data:
                    return data["collection"]
            return data if isinstance(data, list) else []
    except FileNotFoundError:
        print(f"⚠️ 文件不存在: {filepath}", file=sys.stderr, flush=True)
        return []
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析错误 {filepath}: {e}", file=sys.stderr, flush=True)
        return []
    except Exception as e:
        print(f"⚠️ 无法加载 {filepath}: {e}", file=sys.stderr, flush=True)
        return []


def generate_doc_id(source, identifier):
    """生成唯一文档ID"""
    if not identifier:
        identifier = "unknown"
    unique_string = f"{source}_{identifier}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:16]


def unify_pubmed(papers):
    """统一PubMed格式"""
    unified = []
    
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        
        pmid = paper.get("pmid", "") or paper.get("pmid", "")
        if not pmid:
            continue  # 必须有PMID
            
        doc = {
            "doc_id": generate_doc_id("pubmed", pmid),
            "source": "PubMed",
            "source_id": pmid,
            "title": paper.get("title", "").strip(),
            "abstract": paper.get("abstract", "").strip(),
            "authors": paper.get("authors", ""),
            "journal": paper.get("journal", ""),
            "date": paper.get("date", ""),
            "doi": paper.get("doi", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else "",
            "keywords": [],
            "full_text": "",  # PubMed只有摘要
            "methods": "",
            "metadata": {
                "pmid": pmid,
                "pmc_id": paper.get("pmc_id", "")
            }
        }
        unified.append(doc)
    
    return unified


def unify_biorxiv(papers):
    """统一bioRxiv/medRxiv格式"""
    unified = []
    
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        
        doi = paper.get("doi", "")
        if not doi:
            continue  # 必须有DOI
        
        # 判断来源
        doi_lower = doi.lower()
        if "biorxiv" in doi_lower:
            source = "bioRxiv"
        elif "medrxiv" in doi_lower:
            source = "medRxiv"
        else:
            source = "bioRxiv"  # 默认
        
        doc = {
            "doc_id": generate_doc_id("biorxiv", doi),
            "source": source,
            "source_id": doi,
            "title": paper.get("title", "").strip(),
            "abstract": paper.get("abstract", "").strip(),
            "authors": paper.get("authors", ""),
            "journal": "",  # 预印本无期刊
            "date": paper.get("date", ""),
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else "",
            "keywords": [],
            "full_text": "",
            "methods": "",
            "metadata": {
                "category": paper.get("category", ""),
                "version": paper.get("version", "")
            }
        }
        unified.append(doc)
    
    return unified


def unify_pmc(papers):
    """统一PMC格式"""
    unified = []
    
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        
        pmc_id = paper.get("pmc_id", "")
        # 如果没有PMC ID，尝试使用DOI
        if not pmc_id or pmc_id == "N/A":
            pmc_id = paper.get("doi", "")
        
        if not pmc_id or pmc_id == "N/A":
            continue  # 必须有标识符
        
        doc = {
            "doc_id": generate_doc_id("pmc", pmc_id),
            "source": "PMC",
            "source_id": pmc_id,
            "title": paper.get("title", "").strip(),
            "abstract": paper.get("abstract", "").strip(),
            "authors": paper.get("authors", ""),
            "journal": paper.get("journal", ""),
            "date": paper.get("date", ""),
            "doi": paper.get("doi", ""),
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}" if pmc_id and pmc_id != "N/A" else "",
            "keywords": [],
            "full_text": "",
            "methods": paper.get("methods", ""),  # PMC可能有Methods部分
            "metadata": {
                "pmc_id": paper.get("pmc_id", ""),
                "pmid": paper.get("pmid", "")
            }
        }
        unified.append(doc)
    
    return unified


def unify_arxiv(papers):
    """统一arXiv格式"""
    unified = []
    
    for paper in papers:
        doc = {
            "doc_id": generate_doc_id("arxiv", paper.get("arxiv_id", "")),
            "source": "arXiv",
            "source_id": paper.get("arxiv_id", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else paper.get("authors", ""),
            "journal": paper.get("journal_ref", ""),
            "date": paper.get("published", ""),
            "doi": paper.get("doi", ""),
            "url": paper.get("pdf_url", ""),
            "keywords": paper.get("categories", []),
            "full_text": "",
            "methods": "",
            "metadata": {
                "arxiv_id": paper.get("arxiv_id", ""),
                "categories": paper.get("categories", []),
                "primary_category": paper.get("primary_category", ""),
                "comment": paper.get("comment", "")
            }
        }
        unified.append(doc)
    
    return unified


def unify_github(repos):
    """统一GitHub格式"""
    unified = []
    
    for repo in repos:
        doc = {
            "doc_id": generate_doc_id("github", repo.get("full_name", "")),
            "source": "GitHub",
            "source_id": repo.get("full_name", ""),
            "title": repo.get("name", ""),
            "abstract": repo.get("description", ""),
            "authors": "",  # GitHub无明确作者列表
            "journal": "",
            "date": repo.get("created_at", "")[:10] if repo.get("created_at") else "",
            "doi": "",
            "url": repo.get("url", ""),
            "keywords": repo.get("topics", []),
            "full_text": repo.get("readme", ""),
            "methods": "",
            "metadata": {
                "stars": repo.get("stars", 0),
                "forks": repo.get("forks", 0),
                "language": repo.get("language", ""),
                "license": repo.get("license", ""),
                "category": repo.get("category", ""),
                "updated_at": repo.get("updated_at", "")
            }
        }
        unified.append(doc)
    
    return unified




def parse_pubmed_txt(filepath):
    """解析文本格式的PubMed数据"""
    papers = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按论文分割（以数字编号开头）
        paper_blocks = re.split(r'\n(?=\d+\.\s)', content)
        
        for block in paper_blocks:
            if not block.strip():
                continue
            
            paper = {}
            lines = block.split('\n')
            
            # 提取PMID
            pmid_match = re.search(r'PMID:\s*(\d+)', block)
            if pmid_match:
                paper['pmid'] = pmid_match.group(1)
            
            # 提取DOI
            doi_match = re.search(r'doi:\s*([^\s\n]+)', block, re.IGNORECASE)
            if doi_match:
                paper['doi'] = doi_match.group(1)
            
            # 提取标题（通常在编号后的几行，但可能隔着空行）
            first_num_idx = None
            for i, line in enumerate(lines):
                if re.match(r'^\d+\.', line.strip()):
                    first_num_idx = i
                    break
            if first_num_idx is not None:
                # 在编号行之后，寻找第一个非空且不是元数据的行作为标题
                for j in range(first_num_idx + 1, min(first_num_idx + 6, len(lines))):
                    title_line = lines[j].strip()
                    if title_line and not title_line.startswith(('Author', 'DOI', 'PMID', 'PMCID')):
                        paper['title'] = title_line
                        break
            
            # 提取摘要（优先使用带BACKGROUND结构的格式）
            abstract_match = re.search(
                r'BACKGROUND:(.*?)(?:CONCLUSIONS?|DOI|PMID|PMCID|$)',
                block,
                re.DOTALL | re.IGNORECASE
            )
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                abstract = re.sub(r'\s+', ' ', abstract)
                paper['abstract'] = abstract
            else:
                # 回退策略：从 Author information 之后到 DOI/PMID/PMCID 之前的主要段落视作摘要
                lines_stripped = [ln.rstrip() for ln in lines]
                author_idx = None
                for i, ln in enumerate(lines_stripped):
                    if ln.strip().lower().startswith("author information"):
                        author_idx = i
                        break
                
                if author_idx is not None:
                    # 找到作者信息后第一个“非空且前一行为空”的位置作为摘要起点
                    start_idx = None
                    for i in range(author_idx + 1, len(lines_stripped)):
                        if lines_stripped[i].strip() and (i == author_idx + 1 or not lines_stripped[i-1].strip()):
                            start_idx = i
                            break
                    
                    if start_idx is not None:
                        abs_lines = []
                        for j in range(start_idx, len(lines_stripped)):
                            s = lines_stripped[j].strip()
                            # 遇到 DOI/PMID/PMCID 之类的标签则停止
                            if re.match(r'^(DOI:|PMID:|PMCID:)', s, re.IGNORECASE):
                                break
                            if not s:
                                # 允许内部空行，但如果下一行是 DOI/PMID/PMCID 就提前结束
                                if j + 1 < len(lines_stripped) and re.match(
                                    r'^(DOI:|PMID:|PMCID:)', lines_stripped[j+1].strip(), re.IGNORECASE
                                ):
                                    break
                                continue
                            abs_lines.append(s)
                        
                        abstract = re.sub(r'\s+', ' ', " ".join(abs_lines)).strip()
                        if abstract:
                            paper['abstract'] = abstract
            
            # 提取作者（Author information部分）
            author_match = re.search(r'Author information:(.*?)(?:BACKGROUND|DOI|PMID|PMCID|$)', block, re.DOTALL | re.IGNORECASE)
            if author_match:
                author_text = author_match.group(1)
                # 提取作者姓名
                authors = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', author_text)
                if authors:
                    paper['authors'] = ', '.join(authors[:10])  # 最多取前10个作者
            
            # 提取期刊和日期
            journal_match = re.match(r'^\d+\.\s*([^.]+)\.\s*(\d{4})', block)
            if journal_match:
                paper['journal'] = journal_match.group(1).strip()
                paper['date'] = journal_match.group(2)
            
            if paper.get('pmid') or paper.get('doi'):
                papers.append(paper)
    
    except Exception as e:
        print(f"⚠️ 解析PubMed文本文件时出错 {filepath}: {e}", file=sys.stderr, flush=True)
    
    return papers





if __name__ == "__main__":
    
    print("="*60, flush=True)
    print("步骤1: 数据统一与整合", flush=True)
    print("="*60 + "\n", flush=True)
    
    # 创建输出目录
    os.makedirs("Knowledge_Corpus/data/unified", exist_ok=True)
    os.makedirs("Knowledge_Corpus/data/metadata", exist_ok=True)
    
    all_docs = []
    source_stats = defaultdict(int)
    
    # ========== 处理各数据源 ==========
    
    # 1. PubMed
    print("1️⃣ 处理 PubMed 数据...", flush=True)
    try:
        # 处理JSON格式
        pubmed_json_files = [f for f in os.listdir("Knowledge_Corpus/data/raw") if f.startswith("pubmed") and f.endswith(".json")]
        if pubmed_json_files:
            print(f"   找到 {len(pubmed_json_files)} 个JSON文件", flush=True)
            for i, file in enumerate(pubmed_json_files, 1):
                print(f"   处理 ({i}/{len(pubmed_json_files)}): {file}...", end=" ", flush=True)
                papers = load_json_safe(f"Knowledge_Corpus/data/raw/{file}")
                if papers:
                    unified = unify_pubmed(papers)
                    all_docs.extend(unified)
                    source_stats["PubMed"] += len(unified)
                    print(f"✅ {len(unified)} 篇", flush=True)
                else:
                    print(f"⚠️ 无数据", flush=True)
        
        
        print(f"   ✅ PubMed总计: {source_stats['PubMed']} 篇\n", flush=True)
    except Exception as e:
        print(f"   ❌ 处理PubMed时出错: {e}\n", file=sys.stderr, flush=True)
    
    # 2. bioRxiv/medRxiv
    print("2️⃣ 处理 bioRxiv数据...", flush=True)
    try:
        biorxiv_files = [f for f in os.listdir("Knowledge_Corpus/data/raw") if ("biorxiv" in f) and f.endswith(".json")]
        for file in biorxiv_files:
            papers = load_json_safe(f"Knowledge_Corpus/data/raw/{file}")
            if papers:
                unified = unify_biorxiv(papers)
                all_docs.extend(unified)
                for doc in unified:
                    source_stats[doc["source"]] += 1
        print(f"   ✅ bioRxiv: {source_stats.get('bioRxiv', 0)} 篇", flush=True)
    except Exception as e:
        print(f"   ❌ 处理bioRxiv时出错: {e}\n", file=sys.stderr, flush=True)
    
    # 3. PMC
    print("3️⃣ 处理 PMC 数据...", flush=True)
    try:
        pmc_files = [f for f in os.listdir("Knowledge_Corpus/data/raw") if "pmc" in f and f.endswith(".json")]
        for file in pmc_files:
            papers = load_json_safe(f"Knowledge_Corpus/data/raw/{file}")
            if papers:
                unified = unify_pmc(papers)
                all_docs.extend(unified)
                source_stats["PMC"] += len(unified)
        print(f"   ✅ PMC: {source_stats['PMC']} 篇\n", flush=True)
    except Exception as e:
        print(f"   ❌ 处理PMC时出错: {e}\n", file=sys.stderr, flush=True)
    
    # 4. arXiv
    print("4️⃣ 处理 arXiv 数据...", flush=True)
    try:
        arxiv_files = [f for f in os.listdir("Knowledge_Corpus/data/raw") if "arxiv" in f and f.endswith(".json")]
        for file in arxiv_files:
            papers = load_json_safe(f"Knowledge_Corpus/data/raw/{file}")
            if papers:
                unified = unify_arxiv(papers)
                all_docs.extend(unified)
                source_stats["arXiv"] += len(unified)
        print(f"   ✅ arXiv: {source_stats['arXiv']} 篇\n", flush=True)
    except Exception as e:
        print(f"   ❌ 处理arXiv时出错: {e}\n", file=sys.stderr, flush=True)
    
    # 5. GitHub
    print("5️⃣ 处理 GitHub 数据...", flush=True)
    try:
        github_files = [f for f in os.listdir("Knowledge_Corpus/data/raw") if "github" in f and f.endswith(".json")]
        for file in github_files:
            repos = load_json_safe(f"Knowledge_Corpus/data/raw/{file}")
            if repos:
                unified = unify_github(repos)
                all_docs.extend(unified)
                source_stats["GitHub"] += len(unified)
        print(f"   ✅ GitHub: {source_stats['GitHub']} 个仓库\n", flush=True)
    except Exception as e:
        print(f"   ❌ 处理GitHub时出错: {e}\n", file=sys.stderr, flush=True)
    
    
    
    
    # ========== 保存统一格式数据 ==========
    
    # 去重（基于doc_id）
    print(f"{'='*60}", flush=True)
    print(f"去重处理...", flush=True)
    print(f"{'='*60}", flush=True)
    
    seen_ids = set()
    unique_docs = []
    for doc in all_docs:
        doc_id = doc.get("doc_id", "")
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
    
    print(f"原始文档数: {len(all_docs)}", flush=True)
    print(f"去重后文档数: {len(unique_docs)}", flush=True)
    print(f"重复文档数: {len(all_docs) - len(unique_docs)}\n", flush=True)
    
    print(f"{'='*60}", flush=True)
    print(f"✅ 统一格式完成", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    print(f"总文档数: {len(unique_docs)}\n", flush=True)
    
    print("按来源统计:", flush=True)
    for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source:20s}: {count:5d} 条", flush=True)
    
    # 保存
    try:
        with open("Knowledge_Corpus/data/unified/all_documents_raw.json", "w", encoding="utf-8") as f:
            json.dump(unique_docs, f, indent=2, ensure_ascii=False)
        print(f"\n💾 已保存到: Knowledge_Corpus/data/unified/all_documents_raw.json", flush=True)
    except Exception as e:
        print(f"\n❌ 保存文档时出错: {e}", file=sys.stderr, flush=True)
    
    # 保存统计
    try:
        with open("Knowledge_Corpus/data/metadata/01_unify_stats.json", "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(unique_docs),
                "original_count": len(all_docs),
                "duplicates_removed": len(all_docs) - len(unique_docs),
                "source_stats": dict(source_stats)
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 统计已保存到: Knowledge_Corpus/data/metadata/01_unify_stats.json", flush=True)
    except Exception as e:
        print(f"❌ 保存统计时出错: {e}", file=sys.stderr, flush=True)
    
    print(f"\n✅ 完成！", flush=True)
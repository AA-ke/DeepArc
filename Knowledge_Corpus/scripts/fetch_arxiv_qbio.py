import arxiv
import json
import time
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

def fetch_arxiv_by_category(category, max_results=500, days_back=365):
    """
    按分类获取arXiv论文
    
    Args:
        category: arXiv分类代码（如 "q-bio.GN"）
        max_results: 最大结果数
        days_back: 获取最近N天的论文
    """
    
    # 计算日期范围（使用UTC时区，因为arXiv返回的是UTC时间）
    from datetime import timezone
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    print(f"🔍 搜索 {category} (最近 {days_back} 天)...", end=" ", flush=True)
    
    papers = []
    
    try:
        # 使用新的Client API
        client = arxiv.Client()
        
        # 构建查询
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        result_count = 0
        for result in client.results(search):
            # 检查日期范围（确保时区一致）
            if result.published:
                # 如果published是naive datetime，转换为aware
                if result.published.tzinfo is None:
                    published_date = result.published.replace(tzinfo=timezone.utc)
                else:
                    published_date = result.published
                
                if published_date < start_date:
                    break
            
            try:
                paper = {
                    "arxiv_id": result.entry_id.split('/')[-1] if result.entry_id else "N/A",
                    "title": result.title or "N/A",
                    "abstract": result.summary.replace('\n', ' ') if result.summary else "",
                    "authors": [author.name for author in result.authors] if result.authors else [],
                    "published": result.published.strftime("%Y-%m-%d") if result.published else "N/A",
                    "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None,
                    "categories": list(result.categories) if result.categories else [],
                    "primary_category": result.primary_category or "N/A",
                    "pdf_url": result.pdf_url or "",
                    "doi": result.doi or None,
                    "journal_ref": result.journal_ref or None,
                    "comment": result.comment or None,
                    "source": "arXiv"
                }
                papers.append(paper)
                result_count += 1
                
                # 每50篇显示一次进度
                if result_count % 50 == 0:
                    print(f"已获取 {result_count} 篇...", end=" ", flush=True)
                    
            except Exception as e:
                print(f"\n  ⚠️ 解析论文时出错: {e}", file=sys.stderr, flush=True)
                continue
        
        print(f"✅ {len(papers)} 篇", flush=True)
        
    except arxiv.UnexpectedEmptyPageError:
        print(f"✅ {len(papers)} 篇（已到末尾）", flush=True)
    except arxiv.HTTPError as e:
        print(f"❌ HTTP错误: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr, flush=True)
    
    return papers


def save_papers(papers, filename):
    """保存论文"""
    os.makedirs("Knowledge_Corpus/data/raw", exist_ok=True)
    
    # JSON
    json_path = f"Knowledge_Corpus/data/raw/{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    # TXT
    txt_path = f"Knowledge_Corpus/data/raw/{filename}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, paper in enumerate(papers, 1):
            f.write(f"{'='*100}\n")
            f.write(f"Paper {i}/{len(papers)}\n")
            f.write(f"{'='*100}\n")
            f.write(f"arXiv ID: {paper['arxiv_id']}\n")
            f.write(f"Title: {paper['title']}\n")
            f.write(f"Authors: {', '.join(paper['authors'])}\n")
            f.write(f"Published Date: {paper['published']}\n")
            f.write(f"Primary Category: {paper['primary_category']}\n")
            f.write(f"All Categories: {', '.join(paper['categories'])}\n")
            f.write(f"PDF: {paper['pdf_url']}\n")
            if paper['doi']:
                f.write(f"DOI: {paper['doi']}\n")
            if paper['journal_ref']:
                f.write(f"Journal Reference: {paper['journal_ref']}\n")
            if paper['comment']:
                f.write(f"Comment: {paper['comment']}\n")
            f.write(f"\nAbstract:\n{paper['abstract']}\n\n")
    
    print(f"💾 已保存: {json_path}", flush=True)
    print(f"💾 已保存: {txt_path}", flush=True)


if __name__ == "__main__":
    
    print("="*60, flush=True)
    print("arXiv Quantitative Biology (q-bio) 论文采集", flush=True)
    print("="*60 + "\n", flush=True)
    
    # arXiv q-bio 分类（聚焦：基因调控元件设计，兼顾基础与深度）
    qbio_categories = {
         "q-bio.GN": {
            "name": "Genomics (基因组学)",
            "priority": "HIGH",
            "max_results": 400,
            "days_back": 730,  # 2年
            "keywords": [
                # 基础基因调控与元件
                ["gene regulation", "cis-regulatory element", "regulatory element"],
                ["promoter", "enhancer", "silencer", "insulator"],
                ["transcription factor", "TF binding site", "TFBS"],
                ["chromatin accessibility", "ATAC-seq", "DNase-seq"],
                # 调控元件设计与预测
                ["regulatory element design", "cis-regulatory design", "regulatory grammar"],
                ["synthetic enhancer", "synthetic promoter", "synthetic regulatory element"],
                ["MPRA", "massively parallel reporter assay"],
                ["CRE activity prediction", "enhancer activity prediction"],
            ]
        },
        
        "q-bio.QM": {
            "name": "Quantitative Methods (定量方法)",
            "priority": "HIGH",
            "max_results": 400,
            "days_back": 730,
            "keywords": [
                # 用于调控元件设计的定量/机器学习方法
                ["sequence-to-function model", "sequence function prediction"],
                ["deep learning for genomics", "deep learning for gene regulation"],
                ["convolutional neural network", "CNN", "transformer"],
                ["probabilistic model", "Bayesian model", "generative model"],
                # 设计优化与搜索
                ["sequence design", "sequence optimization", "in silico design"],
                ["Bayesian optimization", "evolutionary algorithm", "genetic algorithm"],
                ["inverse design", "optimal regulatory sequence"],
                # 高通量数据建模
                ["MPRA modeling", "massively parallel reporter assay analysis"],
                ["predictive model of enhancer activity", "TF binding prediction"],
            ]
        },
        
        "q-bio.MN": {
            "name": "Molecular Networks (分子网络)",
            "priority": "MEDIUM",
            "max_results": 300,
            "days_back": 730,
            "keywords": [
                ["gene regulatory network", "GRN", "transcriptional network"],
                ["cis-regulatory network", "enhancer-promoter interaction"],
                ["3D genome", "chromatin looping", "chromatin contact"],
                ["network-based design", "network-constrained design"],
                ["systems biology", "pathway-level regulation"],
            ]
        },
        
        "q-bio.BM": {
            "name": "Biomolecules (生物分子)",
            "priority": "MEDIUM",
            "max_results": 300,
            "days_back": 730,
            "keywords": [
                # DNA / RNA 调控序列
                ["regulatory DNA sequence", "regulatory RNA sequence"],
                ["motif discovery", "PWM", "position weight matrix"],
                ["binding motif", "TF binding motif"],
                ["RNA regulatory element", "UTR element"],
                # 结构与功能联系
                ["sequence grammar", "regulatory code"],
                ["biophysical model of binding", "binding energetics"],
            ]
        },
        
        "q-bio.PE": {
            "name": "Populations and Evolution (群体与进化)",
            "priority": "MEDIUM",
            "max_results": 250,
            "days_back": 730,
            "keywords": [
                ["regulatory evolution", "cis-regulatory evolution"],
                ["enhancer evolution", "promoter evolution"],
                ["selection on regulatory elements", "adaptive regulatory change"],
                ["comparative regulatory genomics", "conserved non-coding element"],
            ]
        },
        
        "q-bio.CB": {
            "name": "Cell Behavior (细胞行为)",
            "priority": "LOW",
            "max_results": 150,
            "days_back": 365,
            "keywords": [
                ["cell-type-specific enhancer", "cell type specific regulatory element"],
                ["gene regulation in cell differentiation", "regulatory program"],
                ["single-cell gene regulation", "single-cell regulatory landscape"],
                ["spatial gene regulation", "spatial enhancer activity"],
            ]
        },
        
        "q-bio.NC": {
            "name": "Neurons and Cognition (神经与认知)",
            "priority": "LOW",
            "max_results": 100,
            "days_back": 365,
            "keywords": [
                ["neuronal enhancer", "brain-specific regulatory element"],
                ["gene regulation in neural development", "neurodevelopmental regulation"],
                ["regulatory elements in cognition", "brain regulatory landscape"],
            ]
        },
        
        "q-bio.SC": {
            "name": "Subcellular Processes (亚细胞过程)",
            "priority": "LOW",
            "max_results": 100,
            "days_back": 365,
            "keywords": [
                ["transcription regulation", "transcriptional control"],
                ["promoter architecture", "core promoter elements"],
                ["enhancer-promoter communication", "transcriptional bursting"],
            ]
        },
        
        "q-bio.TO": {
            "name": "Tissues and Organs (组织与器官)",
            "priority": "LOW",
            "max_results": 100,
            "days_back": 365,
            "keywords": [
                ["tissue-specific enhancer", "tissue-specific promoter"],
                ["regulatory element atlas", "regulatory annotation in tissues"],
            ]
        },
        
        "q-bio.OT": {
            "name": "Other Quantitative Biology (其他)",
            "priority": "SKIP",  # 跳过，内容太杂
            "max_results": 0,
            "days_back": 0,
            "keywords": []
        },
    }
    
    all_papers = []
    category_stats = {}
    
    # 按分类获取
    # 过滤掉需要跳过的分类
    active_categories = {k: v for k, v in qbio_categories.items() if v.get('priority') != 'SKIP'}
    total_categories = len(active_categories)
    
    for i, (category, config) in enumerate(active_categories.items(), 1):
        category_name = config.get('name', category)
        priority = config.get('priority', 'MEDIUM')
        max_results = config.get('max_results', 300)
        days_back = config.get('days_back', 730)
        
        print(f"\n[{i}/{total_categories}] [{category}] {category_name} (优先级: {priority})", flush=True)
        
        try:
            papers = fetch_arxiv_by_category(
                category=category,
                max_results=max_results,
                days_back=days_back
            )
            
            if papers:
                # 保存该分类
                cat_name = category.replace(".", "_")
                save_papers(papers, f"arxiv_{cat_name}")
                
                all_papers.extend(papers)
                category_stats[category] = len(papers)
            else:
                print(f"  ⚠️ 未找到论文", flush=True)
                category_stats[category] = 0
            
            time.sleep(3)  # 礼貌延迟
        except Exception as e:
            print(f"  ❌ 处理分类时出错: {e}", file=sys.stderr, flush=True)
            category_stats[category] = 0
            continue
    
    # 去重（一篇论文可能属于多个分类）
    print(f"\n{'='*60}", flush=True)
    print("去重处理...", flush=True)
    print(f"{'='*60}", flush=True)
    
    if not all_papers:
        print("⚠️  未获取到任何论文", flush=True)
        sys.exit(0)
    
    unique_papers = {}
    for paper in all_papers:
        arxiv_id = paper.get('arxiv_id', '')
        if arxiv_id and arxiv_id not in unique_papers:
            unique_papers[arxiv_id] = paper
    
    unique_list = list(unique_papers.values())
    
    print(f"原始总数: {len(all_papers)} 篇", flush=True)
    print(f"去重后: {len(unique_list)} 篇", flush=True)
    
    # 保存所有（去重后）
    if unique_list:
        save_papers(unique_list, "arxiv_qbio_all_unique")
    
    # 统计报告
    print(f"\n📊 统计报告:", flush=True)
    print(f"{'─'*60}", flush=True)
    
    print(f"\n按分类:", flush=True)
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        cat_config = qbio_categories.get(cat, {})
        desc = cat_config.get('name', cat) if isinstance(cat_config, dict) else str(cat_config)
        print(f"  {cat:12s} ({desc:30s}): {count:3d} 篇", flush=True)
    
    # 按年份
    years = defaultdict(int)
    for paper in unique_list:
        published = paper.get('published', '')
        if published and len(published) >= 4:
            year = published[:4]
            if year.isdigit():
                years[year] += 1
    
    if years:
        print(f"\n按年份:", flush=True)
        for year in sorted(years.keys(), reverse=True):
            print(f"  {year}: {years[year]:3d} 篇", flush=True)
    
    # Top作者
    author_count = defaultdict(int)
    for paper in unique_list:
        authors = paper.get('authors', [])
        for author in authors[:3]:  # 只统计前3位作者
            if author:
                author_count[author] += 1
    
    if author_count:
        print(f"\n高产作者 (Top 10):", flush=True)
        for author, count in sorted(author_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {author:40s}: {count:2d} 篇", flush=True)
    
    # 保存元数据
    try:
        metadata = {
            "fetch_date": datetime.now().isoformat(),
            "total_papers": len(unique_list),
            "date_range": {
                "start": min(p.get('published', '') for p in unique_list if p.get('published')),
                "end": max(p.get('published', '') for p in unique_list if p.get('published'))
            },
            "category_stats": category_stats,
            "year_stats": dict(years)
        }
        
        os.makedirs("Knowledge_Corpus/data/metadata", exist_ok=True)
        with open("Knowledge_Corpus/data/metadata/arxiv_qbio_summary.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 元数据已保存: Knowledge_Corpus/data/metadata/arxiv_qbio_summary.json", flush=True)
    except Exception as e:
        print(f"\n⚠️ 保存元数据时出错: {e}", file=sys.stderr, flush=True)
    
    print(f"\n✅ 完成！", flush=True)
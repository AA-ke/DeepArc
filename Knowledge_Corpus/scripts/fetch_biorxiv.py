import requests
import json
import sys
import time
from datetime import datetime, timedelta

def fetch_biorxiv_recent(categories=None, days=30, max_papers=1000):
    """获取最近N天的bioRxiv论文，支持分页和多个类别"""
    if categories is None:
        categories = ["bioinformatics", "computational biology", "genomics", "systems biology"]
    
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        print(f"正在从 bioRxiv API 获取数据 ({start_date} 到 {end_date})...", flush=True)
        print(f"搜索类别: {', '.join(categories)}", flush=True)
        
        all_papers = []
        cursor = 0
        page_size = 100
        
        while len(all_papers) < max_papers:
            url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                papers_batch = data.get("collection", [])
                
                if not papers_batch:
                    print(f"  已获取所有可用论文", flush=True)
                    break
                
                all_papers.extend(papers_batch)
                print(f"  已获取 {len(all_papers)} 篇论文...", flush=True)
                
                # 检查是否还有更多数据
                if len(papers_batch) < page_size:
                    break
                
                cursor += len(papers_batch)
                
                # 避免请求过快
                time.sleep(0.5)
            else:
                print(f"API 请求失败，状态码: {response.status_code}", file=sys.stderr, flush=True)
                break
        
        print(f"API 总共返回 {len(all_papers)} 篇论文", flush=True)
        
        # 筛选相关类别的论文
        if categories:
            papers = []
            for paper in all_papers:
                paper_category = paper.get("category", "").lower()
                if any(cat.lower() in paper_category for cat in categories):
                    papers.append(paper)
            print(f"筛选后找到 {len(papers)} 篇相关类别的论文", flush=True)
        else:
            papers = all_papers
            print(f"未筛选，保留所有 {len(papers)} 篇论文", flush=True)
        
        return papers[:max_papers]
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}", file=sys.stderr, flush=True)
        return []
    except Exception as e:
        print(f"获取数据时出错: {e}", file=sys.stderr, flush=True)
        return []

# 获取数据
print("开始获取 bioRxiv 数据...", flush=True)
# 搜索多个相关类别以获取更多论文
categories = [
    "bioinformatics",
    "computational biology", 
    "genomics",
    "systems biology",
    "genetics",
    "molecular biology"
]
papers = fetch_biorxiv_recent(categories=categories, days=1000, max_papers=2000)

if not papers:
    print("⚠️  未找到论文，请检查网络连接或 API 状态", flush=True)
    sys.exit(1)

print(f"找到 {len(papers)} 篇论文", flush=True)

# 保存元数据
try:
    print("正在保存元数据...", flush=True)
    with open("Knowledge_Corpus/data/metadata/biorxiv_recent.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print("✅ 元数据已保存到 Knowledge_Corpus/data/metadata/biorxiv_recent.json", flush=True)
except Exception as e:
    print(f"保存元数据时出错: {e}", file=sys.stderr, flush=True)

# 提取摘要
try:
    print("正在提取摘要...", flush=True)
    with open("Knowledge_Corpus/data/raw/biorxiv_abstracts.txt", "w", encoding="utf-8") as f:
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "无标题")
            abstract = paper.get("abstract", "无摘要")
            doi = paper.get("doi", "无 DOI")
            
            f.write(f"TITLE: {title}\n")
            f.write(f"ABSTRACT: {abstract}\n")
            f.write(f"DOI: {doi}\n")
            f.write("-" * 80 + "\n\n")
            
            if i % 10 == 0:
                print(f"  已处理 {i}/{len(papers)} 篇论文", flush=True)
    
    print(f"✅ 摘要已保存到 Knowledge_Corpus/data/raw/biorxiv_abstracts.txt", flush=True)
except Exception as e:
    print(f"保存摘要时出错: {e}", file=sys.stderr, flush=True)

print("✅ 完成！检查 Knowledge_Corpus/data/metadata/ 和 Knowledge_Corpus/data/raw/ 目录", flush=True)
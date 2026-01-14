from Bio import Entrez
import json
import time
import os
import sys
from xml.etree import ElementTree as ET
from http.client import IncompleteRead

# ⚠️ 必须设置你的邮箱（NCBI要求）
Entrez.email = "ake0906ake@gmail.com"

def search_pmc_open_access(query, max_results=500):
    """搜索PMC开放获取文章"""
    print(f"🔍 搜索PMC: {query}", flush=True)
    
    try:
        handle = Entrez.esearch(
            db="pmc",
            term=f"{query} AND open access[filter]",
            retmax=max_results,
            sort="relevance",
            usehistory="y"  # 使用历史记录以支持大批量下载
        )
        record = Entrez.read(handle)
        handle.close()
        
        return {
            "ids": record["IdList"],
            "count": int(record["Count"]),
            "webenv": record.get("WebEnv"),
            "query_key": record.get("QueryKey")
        }
    except Exception as e:
        print(f"❌ 搜索错误: {e}", file=sys.stderr, flush=True)
        return {
            "ids": [],
            "count": 0,
            "webenv": None,
            "query_key": None
        }


def fetch_pmc_abstracts_batch(id_list, batch_size=100):
    """批量获取PMC摘要（不是全文，但更快）"""
    all_papers = []
    
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i+batch_size]
        ids_str = ",".join(batch)
        print(f"  获取 {i+1}-{min(i+batch_size, len(id_list))}/{len(id_list)}...", end=" ", flush=True)

        # 增加重试机制，专门处理 IncompleteRead 等网络不稳定错误
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                handle = Entrez.efetch(
                    db="pmc",
                    id=ids_str,
                    rettype="abstract",
                    retmode="xml"
                )

                xml_data = handle.read()
                handle.close()
                root = ET.fromstring(xml_data)

                # 解析XML
                batch_papers = []
                for article in root.findall(".//article"):
                    paper = parse_pmc_article(article)
                    if paper:
                        batch_papers.append(paper)

                all_papers.extend(batch_papers)
                print(f"✅ +{len(batch_papers)} 篇", flush=True)
                time.sleep(0.4)  # 遵守NCBI限制
                break  # 当前 batch 成功，跳出重试循环

            except IncompleteRead as e:
                print(f"\n  ⚠️ IncompleteRead（第 {attempt}/{max_retries} 次尝试）: {e}", file=sys.stderr, flush=True)
                time.sleep(1.0 * attempt)
                if attempt == max_retries:
                    print("  ❌ 多次 IncompleteRead，跳过该批次", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"\n  ❌ 错误: {e}", file=sys.stderr, flush=True)
                break  # 其他错误没必要重试，直接跳过该批次
    
    return all_papers


def extract_text_recursive(elem):
    """递归提取XML元素的所有文本内容"""
    if elem is None:
        return ""
    
    # 获取直接文本
    text_parts = []
    if elem.text:
        text_parts.append(elem.text.strip())
    
    # 递归处理所有子元素
    for child in elem:
        child_text = extract_text_recursive(child)
        if child_text:
            text_parts.append(child_text)
        # 处理子元素后的尾随文本
        if child.tail:
            text_parts.append(child.tail.strip())
    
    return " ".join(text_parts)


def parse_pmc_article(article):
    """解析PMC XML文章"""
    try:
        # 提取标题（递归提取所有文本）
        title_elem = article.find(".//article-title")
        title = extract_text_recursive(title_elem) if title_elem is not None else "N/A"
        if not title or title.strip() == "":
            title = "N/A"
        
        # 提取摘要（递归提取所有文本）
        abstract_elem = article.find(".//abstract")
        abstract = ""
        if abstract_elem is not None:
            # 提取所有段落
            paragraphs = []
            for p in abstract_elem.findall(".//p"):
                p_text = extract_text_recursive(p)
                if p_text:
                    paragraphs.append(p_text)
            
            # 如果没有找到段落，尝试直接提取abstract的所有文本
            if not paragraphs:
                abstract = extract_text_recursive(abstract_elem)
            else:
                abstract = " ".join(paragraphs)
        
        # 提取作者
        authors = []
        for contrib in article.findall(".//contrib[@contrib-type='author']"):
            name_elem = contrib.find(".//name")
            if name_elem is not None:
                surname = name_elem.find("surname")
                given = name_elem.find("given-names")
                if surname is not None:
                    author_name = surname.text
                    if given is not None:
                        author_name = f"{given.text} {author_name}"
                    authors.append(author_name)
        
        # 提取PMC ID
        pmc_id_elem = article.find(".//article-id[@pub-id-type='pmc']")
        pmc_id = pmc_id_elem.text if pmc_id_elem is not None else "N/A"
        
        # 提取DOI
        doi_elem = article.find(".//article-id[@pub-id-type='doi']")
        doi = doi_elem.text if doi_elem is not None else "N/A"
        
        # 提取发表日期
        pub_date = article.find(".//pub-date")
        date = "N/A"
        if pub_date is not None:
            year = pub_date.find("year")
            month = pub_date.find("month")
            day = pub_date.find("day")
            if year is not None:
                date = year.text
                if month is not None:
                    date += f"-{month.text.zfill(2)}"
                    if day is not None:
                        date += f"-{day.text.zfill(2)}"
        
        return {
            "pmc_id": pmc_id,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors),
            "date": date,
            "source": "PMC"
        }
    
    except Exception as e:
        print(f"⚠️ 解析错误: {e}", file=sys.stderr, flush=True)
        return None


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
            f.write(f"论文 {i}/{len(papers)}\n")
            f.write(f"{'='*100}\n")
            f.write(f"PMC ID: {paper.get('pmc_id', 'N/A')}\n")
            f.write(f"DOI: {paper.get('doi', 'N/A')}\n")
            f.write(f"Title: {paper.get('title', 'N/A')}\n")
            f.write(f"Authors: {paper.get('authors', 'N/A')}\n")
            f.write(f"Date: {paper.get('date', 'N/A')}\n")
            f.write(f"\nAbstract:\n{paper.get('abstract', 'N/A')}\n\n")
    
    print(f"💾 已保存: {json_path}", flush=True)
    print(f"💾 已保存: {txt_path}", flush=True)


if __name__ == "__main__":
    

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
    
    all_papers = []
    all_ids = set()  # 去重
    
    print("="*60, flush=True)
    print("开始从 PMC 获取开放获取文章", flush=True)
    print("="*60 + "\n", flush=True)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] 查询: {query}", flush=True)
        
        try:
            # 搜索（每个主题最多100篇）
            search_result = search_pmc_open_access(query, max_results=100)
            print(f"  找到 {search_result['count']} 篇相关文章", flush=True)
            
            # 去重并限制每个主题最多100篇
            new_ids = [pid for pid in search_result['ids'] if pid not in all_ids]
            new_ids = new_ids[:100]  # 确保每个主题最多100篇
            all_ids.update(new_ids)
            
            if new_ids:
                # 获取摘要
                papers = fetch_pmc_abstracts_batch(new_ids)
                all_papers.extend(papers)
                print(f"  新增 {len(papers)} 篇独特论文（本主题限制：最多100篇）", flush=True)
            else:
                print(f"  无新论文（已去重）", flush=True)
            
            time.sleep(1)  # 礼貌延迟
        except Exception as e:
            print(f"  ❌ 处理查询时出错: {e}", file=sys.stderr, flush=True)
            continue
    
    # 保存所有论文
    if all_papers:
        print(f"\n{'='*60}", flush=True)
        print(f"✅ 总共获取 {len(all_papers)} 篇独特的 PMC 论文", flush=True)
        print(f"{'='*60}", flush=True)
        
        save_papers(all_papers, "pmc_open_access_all")
        
        # 统计
        print(f"\n📊 按年份统计:", flush=True)
        years = {}
        for paper in all_papers:
            year = paper.get('date', 'N/A')[:4]
            if year != 'N/A' and year.isdigit():
                years[year] = years.get(year, 0) + 1
        
        if years:
            for year in sorted(years.keys(), reverse=True):
                print(f"  {year}: {years[year]} 篇", flush=True)
        else:
            print("  无有效年份数据", flush=True)
    else:
        print(f"\n⚠️  未获取到任何论文", flush=True)
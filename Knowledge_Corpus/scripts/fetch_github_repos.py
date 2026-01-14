import requests
import json
import time
import os
import sys
from datetime import datetime

# ⚠️ 可选：添加GitHub Token以提高速率限制（从5次/分钟到30次/分钟）
# 获取token: https://github.com/settings/tokens
GITHUB_TOKEN = None  # 设置为 "ghp_your_token_here" 或保持None

def search_github_repos(query, language=None, min_stars=50, max_results=100):
    """搜索GitHub仓库"""
    
    url = "https://api.github.com/search/repositories"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    # 构建查询
    q = query
    if language:
        q += f" language:{language}"
    if min_stars:
        q += f" stars:>={min_stars}"
    
    params = {
        "q": q,
        "sort": "stars",
        "order": "desc",
        "per_page": min(100, max_results)  # GitHub API最多100/页
    }
    
    print(f"🔍 搜索: {q}", flush=True)
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        # 检查速率限制
        if response.status_code == 403:
            rate_limit_reset = response.headers.get('X-RateLimit-Reset')
            if rate_limit_reset:
                wait_time = int(rate_limit_reset) - int(time.time()) + 5
                if wait_time > 0:
                    print(f"  ⚠️ API速率限制，等待 {wait_time} 秒...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"  ⚠️ API速率限制，等待60秒...", flush=True)
                    time.sleep(60)
            else:
                print(f"  ⚠️ API速率限制，等待60秒...", flush=True)
                time.sleep(60)
            return []
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except:
                pass
            print(f"  ❌ {error_msg}", file=sys.stderr, flush=True)
            return []
        
        data = response.json()
        repos = data.get("items", [])
        total_count = data.get("total_count", 0)
        
        print(f"  ✅ 找到 {len(repos)} 个仓库（总共 {total_count} 个）", flush=True)
        return repos
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 网络错误: {e}", file=sys.stderr, flush=True)
        return []
    except Exception as e:
        print(f"  ❌ 错误: {e}", file=sys.stderr, flush=True)
        return []


def fetch_readme(repo_full_name):
    """获取仓库README内容"""
    
    # 尝试常见的README文件名
    readme_names = ["README.md", "README.MD", "readme.md", "Readme.md"]
    
    # 尝试的分支列表
    branches = ["main", "master", "develop"]
    
    for branch in branches:
        for readme_name in readme_names:
            url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{readme_name}"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.text
            except requests.exceptions.RequestException:
                continue
    
    return None


def process_repo(repo):
    """处理单个仓库，提取关键信息"""
    
    full_name = repo["full_name"]
    print(f"  📦 处理: {full_name}", flush=True)
    
    try:
        # 基本信息
        repo_data = {
            "name": repo.get("name", ""),
            "full_name": full_name,
            "description": repo.get("description", ""),
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "language": repo.get("language", "N/A"),
            "url": repo.get("html_url", ""),
            "topics": repo.get("topics", []),
            "created_at": repo.get("created_at", ""),
            "updated_at": repo.get("updated_at", ""),
            "homepage": repo.get("homepage", ""),
            "license": repo.get("license", {}).get("name", "N/A") if repo.get("license") else "N/A",
            "readme": None,
            "readme_length": 0
        }
        
        # 获取README
        readme = fetch_readme(full_name)
        if readme:
            repo_data["readme"] = readme
            repo_data["readme_length"] = len(readme)
            print(f"    ✅ README: {len(readme)} 字符", flush=True)
        else:
            print(f"    ⚠️ 未找到README", flush=True)
        
        time.sleep(0.5)  # 避免被限速
        
        return repo_data
    except Exception as e:
        print(f"    ❌ 处理仓库时出错: {e}", file=sys.stderr, flush=True)
        return None


def save_repos(repos, filename):
    """保存仓库数据"""
    os.makedirs("Knowledge_Corpus/data/raw", exist_ok=True)
    
    # JSON（完整数据，包括README）
    json_path = f"Knowledge_Corpus/data/raw/{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(repos, f, indent=2, ensure_ascii=False)
    
    # TXT（只保存README内容）
    txt_path = f"Knowledge_Corpus/data/raw/{filename}_readmes.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, repo in enumerate(repos, 1):
            f.write(f"{'='*100}\n")
            f.write(f"Repository {i}/{len(repos)}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Name: {repo['full_name']}\n")
            f.write(f"Description: {repo['description']}\n")
            f.write(f"⭐ Stars: {repo['stars']} | Forks: {repo['forks']}\n")
            f.write(f"Language: {repo['language']}\n")
            f.write(f"Topics: {', '.join(repo['topics'])}\n")
            f.write(f"URL: {repo['url']}\n")
            f.write(f"\n{'─'*100}\n")
            f.write(f"README:\n")
            f.write(f"{'─'*100}\n")
            if repo['readme']:
                f.write(repo['readme'])
            else:
                f.write("No README file found")
            f.write(f"\n\n")
    
    print(f"💾 已保存: {json_path}", flush=True)
    print(f"💾 已保存: {txt_path}", flush=True)


if __name__ == "__main__":
    
    print("="*60, flush=True)
    print("开始从 GitHub 获取生物信息学仓库", flush=True)
    print("="*60 + "\n", flush=True)
    
    # 定义搜索主题
    search_topics = [
         # 通用（最重要）
    {"query": "gene regulatory element", "language": "python", "min_stars": 10, "category": "general"},
    {"query": "cis-regulatory element", "language": None, "min_stars": 10, "category": "general"},
    {"query": "regulatory element design", "language": None, "min_stars": 10, "category": "general"},
    
    # 核心调控元件类型
    {"query": "promoter prediction", "language": "python", "min_stars": 20, "category": "regulatory_elements"},
    {"query": "enhancer prediction", "language": "python", "min_stars": 20, "category": "regulatory_elements"},
    {"query": "synthetic promoter", "language": None, "min_stars": 20, "category": "regulatory_elements"},
    {"query": "synthetic enhancer", "language": None, "min_stars": 20, "category": "regulatory_elements"},
    
    # 计算方法
    {"query": "MPRA", "language": None, "min_stars": 20, "category": "methods"},
    {"query": "STARR-seq", "language": None, "min_stars": 20, "category": "methods"},
    {"query": "regulatory sequence design", "language": "python", "min_stars": 20, "category": "methods"},
    {"query": "sequence-to-function", "language": "python", "min_stars": 20, "category": "methods"},
    
    # 机器学习/深度学习
    {"query": "deep learning gene regulation", "language": "python", "min_stars": 30, "category": "ml"},
    {"query": "neural network enhancer", "language": "python", "min_stars": 20, "category": "ml"},
    {"query": "transformer gene regulation", "language": "python", "min_stars": 20, "category": "ml"},
    {"query": "CNN promoter enhancer", "language": "python", "min_stars": 20, "category": "ml"},
    
    # 转录因子与motif
    {"query": "transcription factor binding", "language": "python", "min_stars": 30, "category": "tf_motif"},
    {"query": "motif discovery", "language": "python", "min_stars": 25, "category": "tf_motif"},
    {"query": "PWM", "language": None, "min_stars": 15, "category": "tf_motif"},
    {"query": "ChIP-seq analysis", "language": "python", "min_stars": 30, "category": "tf_motif"},
    
    # 表观遗传与染色质
    {"query": "ATAC-seq", "language": "python", "min_stars": 30, "category": "epigenomics"},
    {"query": "chromatin accessibility", "language": "python", "min_stars": 25, "category": "epigenomics"},
    {"query": "3D genome", "language": None, "min_stars": 20, "category": "epigenomics"},
    {"query": "chromatin interaction", "language": None, "min_stars": 20, "category": "epigenomics"},
    
    # 基因调控网络
    {"query": "gene regulatory network", "language": "python", "min_stars": 20, "category": "networks"},
    {"query": "GRN inference", "language": "python", "min_stars": 20, "category": "networks"},
    ]
    
    all_repos = []
    seen_repos = set()  # 去重
    
    for i, topic in enumerate(search_topics, 1):
        print(f"\n[{i}/{len(search_topics)}] 主题: {topic['query']}", flush=True)
        
        try:
            # 搜索
            repos = search_github_repos(
                query=topic["query"],
                language=topic.get("language"),
                min_stars=topic["min_stars"],
                max_results=50
            )
            
            if not repos:
                print(f"  跳过：未找到仓库", flush=True)
                time.sleep(1)
                continue
            
            # 处理每个仓库
            processed_count = 0
            for repo in repos:
                full_name = repo.get("full_name")
                if not full_name:
                    continue
                
                # 去重
                if full_name in seen_repos:
                    continue
                seen_repos.add(full_name)
                
                # 获取详细信息
                repo_data = process_repo(repo)
                if repo_data:
                    all_repos.append(repo_data)
                    processed_count += 1
            
            print(f"  本主题新增: {processed_count} 个，累计: {len(all_repos)} 个独特仓库", flush=True)
            
            # 礼貌延迟
            time.sleep(2)
        except Exception as e:
            print(f"  ❌ 处理主题时出错: {e}", file=sys.stderr, flush=True)
            continue
    
    # 保存
    if all_repos:
        print(f"\n{'='*60}", flush=True)
        print(f"✅ 总共获取 {len(all_repos)} 个 GitHub 仓库", flush=True)
        print(f"{'='*60}", flush=True)
        
        save_repos(all_repos, "github_repos_all")
        
        # 统计
        print(f"\n📊 统计信息:", flush=True)
        
        # 按语言
        languages = {}
        for repo in all_repos:
            lang = repo.get('language', 'N/A')
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"\n编程语言:", flush=True)
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {lang}: {count} 个", flush=True)
        
        # 按星标
        total_stars = sum(repo.get('stars', 0) for repo in all_repos)
        avg_stars = total_stars / len(all_repos) if all_repos else 0
        print(f"\n⭐ 总星标数: {total_stars:,}", flush=True)
        print(f"⭐ 平均星标: {avg_stars:.1f}", flush=True)
        
        # 有README的比例
        with_readme = sum(1 for repo in all_repos if repo.get('readme'))
        readme_percent = (with_readme / len(all_repos) * 100) if all_repos else 0
        print(f"\n📄 包含README: {with_readme}/{len(all_repos)} ({readme_percent:.1f}%)", flush=True)
    else:
        print(f"\n⚠️  未获取到任何仓库", flush=True)
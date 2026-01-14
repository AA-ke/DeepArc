"""
RAG/plot_comparison.py
生成RAG vs 无RAG的数据对比图
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# 数据
metrics = [
    'Semantic\nSimilarity',
    'BLEU-4',
    'ROUGE-L',
    'Edit Distance\nSimilarity',
    'Jaccard',
    'Char\nSimilarity'
]

rag_scores = [0.926, 0.043, 0.434, 0.312, 0.245, 0.831]
no_rag_scores = [0.910, 0.021, 0.381, 0.259, 0.181, 0.783]
differences = [0.016, 0.022, 0.053, 0.053, 0.065, 0.048]

# 创建图表
fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# 左图：柱状图对比（统一坐标轴）
x = np.arange(len(metrics))
width = 0.35

# 使用原始值绘制柱状图（No RAG在左边，RAG在右边）
bars1 = ax1.bar(x - width/2, no_rag_scores, width, label='No RAG', 
                color='#E24A4A', alpha=0.85, edgecolor='#A82E2E', linewidth=2)
bars2 = ax1.bar(x + width/2, rag_scores, width, label='RAG', 
                color='#4A90E2', alpha=0.85, edgecolor='#2C5F8F', linewidth=2)

# 添加数值标签
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()  # No RAG
    height2 = bar2.get_height()   # RAG
    # No RAG标签（左边）
    ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
             f'{height1:.3f}', ha='center', va='bottom', fontsize=9, 
             fontweight='bold', color='#A82E2E')
    # RAG标签（右边）
    ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
             f'{height2:.3f}', ha='center', va='bottom', fontsize=9, 
             fontweight='bold', color='#2C5F8F')

ax1.set_xlabel('Evaluation Metrics', fontsize=13, fontweight='bold')
ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
ax1.set_title('RAG vs No RAG: Answer Correctness Comparison', 
              fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10)
ax1.legend(loc='upper left', fontsize=12, framealpha=0.95, shadow=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_ylim([0, max(max(rag_scores), max(no_rag_scores)) * 1.2])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 右图：差异条形图
colors = ['#2ECC71' if d > 0 else '#E74C3C' for d in differences]
bars3 = ax2.barh(metrics, differences, color=colors, alpha=0.85, 
                 edgecolor='#1E8449', linewidth=2)

# 添加数值标签和百分比
for i, (bar, diff) in enumerate(zip(bars3, differences)):
    width = bar.get_width()
    # 计算提升百分比
    improvement_pct = (diff / no_rag_scores[i]) * 100 if no_rag_scores[i] > 0 else 0
    label_text = f'+{diff:.3f} (+{improvement_pct:.1f}%)'
    ax2.text(width + 0.002, bar.get_y() + bar.get_height()/2.,
             label_text, ha='left', va='center', fontsize=10, 
             fontweight='bold', color='#1E8449')

ax2.set_xlabel('Improvement (RAG - No RAG)', fontsize=13, fontweight='bold')
ax2.set_title('RAG Performance Improvement', fontsize=15, fontweight='bold', pad=20)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_xlim([-0.005, max(differences) * 1.25])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 添加总标题
fig.suptitle('RAG System Evaluation Results', 
              fontsize=16, fontweight='bold', y=0.98)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
output_path = Path(__file__).parent / 'rag_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.2)
print(f"Chart saved to: {output_path}")

# 也保存为PDF格式（矢量图）
pdf_path = Path(__file__).parent / 'rag_comparison.pdf'
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.2)
print(f"PDF saved to: {pdf_path}")

# 显示图片
plt.show()


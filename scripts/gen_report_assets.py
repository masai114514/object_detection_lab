#!/usr/bin/env python3
"""
生成 LaTeX 报告所需的图表与表格素材（从训练产物与评估结果自动刷新）。

产出（都在 report/ 下）：
- figures/            训练曲线、混淆矩阵、批次预览、错误案例（统一命名拷贝）
- tables/test_results.tex   20 张个人测试集逐张结果（longtable）
- tables/summary.tex        按类别准确率 + 混淆统计（三线表）
- gitlog.txt                git 增量提交记录（附录用）

用法（在仓库根目录）：
    python3 scripts/gen_report_assets.py
"""
import csv
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(ROOT, 'runs', 'detect', 'combined_model_v3')
FIG_DST = os.path.join(ROOT, 'report', 'figures')
TBL_DST = os.path.join(ROOT, 'report', 'tables')
os.makedirs(FIG_DST, exist_ok=True)
os.makedirs(TBL_DST, exist_ok=True)

# (源, 目标名) —— 统一成不含下划线/大写的名字，方便在模板里 \includegraphics
FIGURES = [
    (os.path.join(RUNDIR, 'results.png'),                 'train_curves.png'),
    (os.path.join(RUNDIR, 'BoxPR_curve.png'),             'pr_curve.png'),
    (os.path.join(RUNDIR, 'BoxF1_curve.png'),             'f1_curve.png'),
    (os.path.join(RUNDIR, 'confusion_matrix_normalized.png'), 'confusion_matrix.png'),
    (os.path.join(RUNDIR, 'labels.jpg'),                  'labels_distribution.jpg'),
    (os.path.join(RUNDIR, 'train_batch0.jpg'),            'train_batch.jpg'),
    (os.path.join(RUNDIR, 'val_batch0_pred.jpg'),         'val_pred.jpg'),
]
ERROR_CASE_SRC = os.path.join(ROOT, 'results', 'error_cases')


def copy_figures():
    for src, name in FIGURES:
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(FIG_DST, name))
            print(f'图: {name}')
        else:
            print(f'!! 缺少 {src}')
    # 错误案例（文件名随机，取 error_cases 里第一个）
    if os.path.isdir(ERROR_CASE_SRC):
        for f in sorted(os.listdir(ERROR_CASE_SRC)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(os.path.join(ERROR_CASE_SRC, f),
                             os.path.join(FIG_DST, 'error_case.jpg'))
                print(f'图: error_case.jpg ({f})')
                break


def gen_tables():
    csv_path = os.path.join(ROOT, 'results', 'test_results.csv')
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # 逐张结果表（longtable，跨页）
    with open(os.path.join(TBL_DST, 'test_results.tex'), 'w') as f:
        f.write("%% 由 scripts/gen_report_assets.py 从 results/test_results.csv 自动生成，勿手改\n")
        f.write(r"\begin{longtable}{r l l l c c}" + "\n")
        f.write(r"\caption{个人测试集逐张结果（20 张，10 cup + 10 mouse，阈值 conf=0.25）}"
                + r"\label{tab:test_results}" + "\\\\\n")
        f.write(r"\toprule 编号 & 图片 & 真值 & 预测 & 置信度 & 正确 \\" + "\n")
        f.write(r"\midrule \endhead" + "\n")
        f.write(r"\bottomrule \endlastfoot" + "\n")
        for i, r in enumerate(rows, 1):
            mark = r"\checkmark" if r['correct'] == 'True' else r"\ensuremath{\times}"
            ok = 'cup' if r['true_class'] == 'cup' else 'mouse'
            f.write(f"{i} & \\texttt{{{r['image']}}} & {ok} & {r['pred_class']} "
                    f"& {r['confidence']} & {mark} \\\\\n")
        f.write(r"\end{longtable}" + "\n")

    # 按类别汇总 + 混淆
    from collections import defaultdict
    per_cls = defaultdict(lambda: [0, 0])          # cls -> [correct, total]
    conf = defaultdict(int)                         # (true, pred)
    confs = defaultdict(list)                       # cls -> 该类全部样本的置信度
    for r in rows:
        t, p = r['true_class'], r['pred_class']
        per_cls[t][1] += 1
        conf[(t, p)] += 1
        confs[t].append(float(r['confidence']))
        if r['correct'] == 'True':
            per_cls[t][0] += 1

    with open(os.path.join(TBL_DST, 'summary.tex'), 'w') as f:
        f.write("%% 由 scripts/gen_report_assets.py 生成\n")
        f.write(r"\begin{table}[htbp]" + "\n\\centering\n")
        f.write(r"\caption{个人测试集按类别准确率（验收要求整体 $\geq 80\%$）}" + "\n")
        f.write(r"\begin{tabular}{l c c c c}" + "\n\\toprule\n")
        f.write("类别 & 测试数 & 正确 & 正确率 & 平均置信度 \\\\\n\\midrule\n")
        total = tot_correct = 0
        for cls in ['cup', 'mouse']:
            c, n = per_cls[cls]
            avg = sum(confs[cls]) / len(confs[cls]) if confs[cls] else 0
            total += n
            tot_correct += c
            pct = f"{c/n:.0%}".replace('%', r'\%')
            f.write(f"{cls} & {n} & {c} & {pct} & {avg:.3f} \\\\\n")
        pct_all = f"{tot_correct/total:.0%}".replace('%', r'\%')
        f.write(f"\\midrule 总体 & {total} & {tot_correct} "
                f"& {pct_all} & -- \\\\\n\\bottomrule\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n\n")
        f.write("%% 混淆情况（行=真值，列=预测）\n")
        f.write(r"\begin{table}[htbp]" + "\n\\centering\n")
        f.write(r"\caption{混淆矩阵（行：真值，列：预测，none=未检出）}" + "\n")
        f.write(r"\begin{tabular}{l c c c}" + "\n\\toprule\n")
        f.write("真值$\\backslash$预测 & cup & mouse & none \\\\\n\\midrule\n")
        for t in ['cup', 'mouse']:
            f.write(f"{t} & {conf[(t,'cup')]} & {conf[(t,'mouse')]} "
                    f"& {conf[(t,'none')]} \\\\\n")
        f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
    print('表: test_results.tex / summary.tex')


def gen_gitlog():
    out = subprocess.run(
        ['git', 'log', '--oneline', '--reverse'],
        capture_output=True, text=True, cwd=ROOT,
    )
    with open(os.path.join(ROOT, 'report', 'gitlog.txt'), 'w') as f:
        f.write(out.stdout)
    print(f"gitlog.txt（{len(out.stdout.splitlines())} 条提交）")


if __name__ == '__main__':
    copy_figures()
    gen_tables()
    gen_gitlog()
    print('完成。在 report/ 目录执行: xelatex report.tex（编译两次生成目录）')

"""
t3_g7_package.py — 阶段 D rev2 / G7：归档 + 数据血缘记录（data_lineage.json）

产出（写到投稿冲刺工作目录根）：
  data_lineage.json      每一条"原始来源 → 派生工件"的血缘链，含 sha256
  version_manifest.csv   工件清单：路径 / 角色 / sha256 / 大小 / 来源包
  SHA256SUMS.txt         整个投稿冲刺目录的逐文件 sha256（可复核）

不改任何结果文件、不重跑任何模型。
"""
import json
import os
import hashlib
import datetime as dt

SPRINT = ARCH_ROOT + '/GeOAD_Submission_Sprint_20261029'
TIER3 = os.path.join(SPRINT, 'tier3')
RQ1 = ARCH_ROOT + '/GeOAD_RQ1_FormalValidation_20261025'
PLANNING = ARCH_ROOT + '/GeOAD_ReEval_Planning_20261028'
T1 = ARCH_ROOT + '/GeOAD_ReEval_Run_20261028_Tier1'
T2 = ARCH_ROOT + '/GeOAD_ReEval_Run_20261028_Tier2'
SEGCONF = ARCH_ROOT + '/GeOAD_SegConfFix_20261027'
RQ2 = ARCH_ROOT + '/GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def entry(path, role, source_pkg, note=''):
    return dict(path=path, role=role, source_package=source_pkg,
                sha256=sha256_file(path), size_bytes=os.path.getsize(path), note=note)


def main():
    master = os.path.join(PLANNING, 'data', 'master_v2_consistent.csv')
    lock = os.path.join(T1, 'protocol', 'protocol_lock.json')

    lineage = {
        'generated_by': 'GeOAD_Submission_Sprint_20261029 / t3_g7_package.py',
        'generated_at': dt.datetime.now().isoformat(timespec='seconds'),
        'protocol_id': 'RQ2-S2-LARGE-DEV-CONFIRM-v1 + stage-C data correction (口径B)',
        'primary_data_line': [
            entry(master, 'authoritative master table (all models trained/evaluated on this)',
                  'GeOAD_ReEval_Planning_20261028',
                  'derived from the stage-C column-level correction applied to the frozen '
                  'RQ2 manifest; 43,834 rows; fit 29,941 / calib 6,411 / dev_confirm 6,417 / '
                  'sealed 1,065 (sealed labels never read)'),
            entry(lock, 'protocol lock (splits, budgets, seeds, checkpoint rule)',
                  'GeOAD_ReEval_Run_20261028_Tier1'),
            entry(os.path.join(SEGCONF, 'data', 'master_v2.csv'),
                  'stage-C corrected master (historical version, superseded by the '
                  'authoritative consistent master)', 'GeOAD_SegConfFix_20261027',
                  'column-level replacement of INTERNAL_INSPECTION_IND (274 rows); '
                  'kept because the baseline inputs were built from it'),
            entry(os.path.join(RQ2, 'protocol', 'data_row_manifest.csv'),
                  'frozen RQ2 manifest (base of the column-level correction)',
                  'GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025'),
        ],
        'baseline_input_line': [
            entry(os.path.join(TIER3, 'data', 'baseline_data_13d', 'X_train.csv'),
                  'external-baseline training features (13 dims, fit split)', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'data', 'baseline_data_13d', 'X_val.csv'),
                  'external-baseline calibration features', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'data', 'baseline_data_13d', 'X_test.csv'),
                  'external-baseline development-confirmation features', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'data', 'baseline_data_13d', 'fair_manifest.json'),
                  'baseline input manifest (vocabulary, file hashes, provenance)',
                  'Tier3 sprint',
                  'PROVENANCE NOTE: its source_sha256 field records '
                  'f007e457... (stage-C master_v2.csv), not the authoritative '
                  'd9ef45f2... (master_v2_consistent.csv). The two differ on the '
                  'development pool only in the derived pattern13 column and in the '
                  'sealed-pool placeholder columns (y / label); the 13 model input '
                  'columns and the labels are identical row for row. The baseline '
                  'inputs therefore required no rebuild; this note records the '
                  'bookkeeping difference so that the lineage is auditable.'),
        ],
        'code_line': [
            entry(os.path.join(TIER3, 'code', 'baselines_fixed', 'MFF-CNN-LightGBM', 'train.py'),
                  'MFF-CNN-LightGBM baseline, objective repaired (six-way softmax + '
                  'sparse categorical cross-entropy)', 'Tier3 sprint',
                  'pre-repair sha256 f576f61853c81fb1d0349c8cacf826e248c3876049524a1035f6501aae77224f; '
                  'post-repair sha256 a100af58be8f0542b63cc9bfe3f53b4b9c8114d3413ba487067b392ce7e65530; '
                  'diff: GeOAD_ReEval_Planning_20261028/code/patch/mff_6class_fix.diff'),
            entry(os.path.join(TIER3, 'code', 'rq1f_run_baselines.py'),
                  'external-baseline runner (Tier 3)', 'Tier3 sprint',
                  'one documented change vs the RQ1 original: the accident_group column is '
                  'read from the authoritative master (d9ef45f2...) instead of the frozen '
                  'RQ2 manifest (c6bcde43...); accident_group is row-identical between the two'),
            entry(os.path.join(TIER3, 'code', 'rq1f_precheck_tier3.py'),
                  'Tier 3 pre-run legality check (36/36 PASS)', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'code', 't3_score.py'),
                  'G5 same-protocol statistics for the eight external baselines', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'code', 't3_g6_mechanism.py'),
                  'G6 knock-on recomputation (pattern coverage, near-duplicate mechanism, '
                  'decision decomposition)', 'Tier3 sprint'),
            entry(os.path.join(TIER3, '_hash_before_after', 'precheck_tier3.log'),
                  'Tier 3 pre-run legality check output (36/36 PASS)', 'Tier3 sprint'),
            entry(os.path.join(TIER3, '_hash_before_after', 'tier3_run.log'),
                  'eight-baseline training log (all statuses RUN)', 'Tier3 sprint'),
            entry(os.path.join(TIER3, '_hash_before_after', 't3_score.log'),
                  'G5 statistics log', 'Tier3 sprint'),
            entry(os.path.join(TIER3, '_hash_before_after', 'mff_train_py.after.sha256'),
                  'post-repair hash of the MFF-CNN-LightGBM training script', 'Tier3 sprint'),
            entry(os.path.join(TIER3, '_hash_before_after', 'mff_train_py.before.sha256'),
                  'pre-repair hash of the MFF-CNN-LightGBM training script', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'mff_fix', 'before', 'train.py'),
                  'MFF-CNN-LightGBM training script BEFORE the feature-packing fix '
                  '(also contains the earlier 6-class objective repair)',
                  'Tier3 sprint',
                  'sha256 a100af58be8f0542b63cc9bfe3f53b4b9c8114d3413ba487067b392ce7e65530; '
                  'packing bug: features_per_pt = n_features // Ncpt = 13//7 = 1, so feature '
                  'indices 7..12 never entered the CNN'),
            entry(os.path.join(TIER3, 'mff_fix', 'train_fixed.py'),
                  'MFF-CNN-LightGBM training script AFTER the feature-packing fix '
                  '(adopted as code/baselines_fixed/MFF-CNN-LightGBM/train.py)',
                  'Tier3 sprint',
                  'one-line change: n_features // Ncpt -> ceil(n_features / Ncpt); '
                  'input shape (n,7,100,1), network, training budget and all other '
                  'configuration unchanged'),
            entry(os.path.join(TIER3, 'mff_fix', 'mff_packing_fix.diff'),
                  'unified diff of the feature-packing fix', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'mff_fix', 'tests', 'test_mff_input_coverage.py'),
                  'per-dimension input coverage test (packing / real-data / network-gradient)',
                  'Tier3 sprint'),
            entry(os.path.join(TIER3, 'mff_fix', 'tests', 'test_mff_trained_model_e2e.py'),
                  'end-to-end per-dimension influence test on the trained model', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'mff_fix', 'logs', 'coverage_test_BEFORE.json'),
                  'coverage test on the pre-fix implementation (FAIL: dims 7-12 uncovered)',
                  'Tier3 sprint'),
            entry(os.path.join(TIER3, 'mff_fix', 'logs', 'coverage_test_AFTER.json'),
                  'coverage test on the fixed implementation (PASS: all 13 dims covered)',
                  'Tier3 sprint'),
            entry(os.path.join(TIER3, 'mff_fix', 'logs', 'e2e_trained_model.json'),
                  'end-to-end trained-model test (PASS: all 13 dims change CNN features, '
                  'LightGBM probabilities and final decisions; predictions reproduce the '
                  'official artifact)', 'Tier3 sprint'),
            entry(os.path.join(TIER3, 'mff_fix', 'before', 'predictions',
                               'MFF-CNN-LightGBM.csv'),
                  'MFF per-sample predictions BEFORE the feature-packing fix '
                  '(Acc 66.2927 / Macro-F1 50.6378)', 'Tier3 sprint'),
            entry(os.path.join(TIER3, '_hash_before_after', 'baseline_code_unchanged.txt'),
                  'file-by-file sha256 comparison of the eight baseline code trees against the '
                  'archived RQ1 originals; shows that only MFF-CNN-LightGBM/train.py differs',
                  'Tier3 sprint'),
        ],
        'result_line': [],
        'manuscript_line': [],
        'sealed_pool': {
            'rows': 1065,
            'definition': 'incident_year >= 2024',
            'action': 'labels never read by any code in this study; excluded from training, '
                      'from model selection and from every reported number',
            'claim_limit': 'not described as "untouched": the corpus was reconstructed and '
                           'audited in earlier rounds before the pool was set aside',
        },
        'superseded_results': {
            'note': 'Results of the 2026-10-24 protocol (attribute-pattern grouping, '
                    'pre-correction data, 6,314-row evaluation split) are NOT used in the '
                    'main text of the manuscript. The only place they appear is Appendix D '
                    'of the manuscript, labelled as superseded.',
        },
    }

    for name, role in [('t3_baseline_metrics.csv', 'eight external baselines, main metrics'),
                       ('t3_baseline_per_class.csv', 'eight external baselines, per-class P/R/F1'),
                       ('t3_baseline_confusion.csv', 'eight external baselines, confusion matrices'),
                       ('t3_baseline_vs_geoad.csv', 'paired comparisons vs the official model'),
                       ('t3_holm.csv', 'Holm correction, per metric family'),
                       ('t3_macro_permutation.csv', 'Macro-F1 accident-group permutation detail'),
                       ('t3_new_vs_old.csv', 'new (corrected data) vs old (10-25) baseline results'),
                       ('t3_g6_mechanism.json', 'G6 knock-on recomputation'),
                       ('t3_score_meta.json', 'G5 metadata: inputs, hashes, protocol')]:
        p = os.path.join(TIER3, 'results', name)
        if os.path.isfile(p):
            lineage['result_line'].append(entry(p, role, 'Tier3 sprint'))

    # ---- manuscript / RQ3 artifacts ----------------------------------------
    paper_items = [
        (os.path.join(SPRINT, 'paper', 'paper_draft_v2.md'),
         'English manuscript master (all numbers re-derived from frozen artifacts)'),
        (os.path.join(SPRINT, 'paper', 'paper.tex'),
         'LaTeX manuscript entry point (MDPI class; not compile-verified on this machine)'),
        (os.path.join(SPRINT, 'paper', 'paper_body.tex'), 'generated LaTeX body'),
        (os.path.join(SPRINT, 'paper', 'paper_appendix.tex'), 'generated LaTeX appendix'),
        (os.path.join(SPRINT, 'paper', 'references.bib'),
         'BibTeX bibliography (every entry verified live against Crossref/DataCite)'),
        (os.path.join(SPRINT, 'paper', 'references_verification.csv'),
         'per-entry verification record (source, HTTP status, notes)'),
        (os.path.join(SPRINT, 'paper', 'references_notes.md'),
         'human-readable reference verification notes'),
        (os.path.join(SPRINT, 'paper', 'build_refs.py'), 'bibliography builder / verifier'),
        (os.path.join(SPRINT, 'paper', 'build_paper.py'), 'deterministic LaTeX project builder'),
        (os.path.join(SPRINT, 'paper', 'md2latex.py'), 'markdown -> LaTeX converter'),
        (os.path.join(SPRINT, 'paper', 'insert_citations.py'), 'citation insertion map'),
        (os.path.join(SPRINT, 'paper', 'audit_paper_numbers.py'),
         're-derives every manuscript number from the frozen artifacts (126 checks)'),
        (os.path.join(SPRINT, 'paper', 'check_latex.py'),
         'static LaTeX checker (environments, braces, refs, cites, table columns, CJK)'),
        (os.path.join(SPRINT, 'rq3', 'results', 'shap_summary.json'),
         'RQ3 TreeSHAP summary (backbone only)'),
        (os.path.join(SPRINT, 'rq3', 'results', 'aux_summary.json'),
         'RQ3 decision-change decomposition (83 / 39 / 30 / 14)'),
        (os.path.join(SPRINT, 'rq3', 'results', 'aux_per_class_corrections.csv'),
         'RQ3 per-class corrections and breakages'),
        (os.path.join(SPRINT, 'rq3', 'results', 'assertions_report.json'),
         'RQ3 self-check report'),
        (os.path.join(SPRINT, 'SUBMISSION_READINESS_REPORT.md'),
         'blocking-issues report'),
    ]
    for p_, role in paper_items:
        if os.path.isfile(p_):
            lineage['manuscript_line'].append(entry(p_, role, 'submission sprint'))
    for fn in sorted(os.listdir(os.path.join(SPRINT, 'paper', 'figures'))):
        p_ = os.path.join(SPRINT, 'paper', 'figures', fn)
        if os.path.isfile(p_):
            lineage['manuscript_line'].append(
                entry(p_, 'manuscript figure (paper/figures)', 'submission sprint'))

    with open(os.path.join(SPRINT, 'data_lineage.json'), 'w', encoding='utf-8') as f:
        json.dump(lineage, f, ensure_ascii=False, indent=2)

    # 工件清单
    rows = []
    for line in (lineage['primary_data_line'] + lineage['baseline_input_line']
                 + lineage['code_line'] + lineage['result_line']
                 + lineage['manuscript_line']):
        rows.append(line)
    import csv
    with open(os.path.join(SPRINT, 'version_manifest.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'role', 'source_package', 'sha256',
                                          'size_bytes', 'note'])
        w.writeheader()
        for r in rows:
            r = dict(r)
            r['note'] = r.get('note', '').replace('\n', ' ')
            w.writerow(r)

    # 全目录 SHA256SUMS
    sums = os.path.join(SPRINT, 'SHA256SUMS.txt')
    with open(sums, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(SPRINT):
            dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
            for fn in sorted(files):
                p = os.path.join(root, fn)
                if os.path.abspath(p) == os.path.abspath(sums):
                    continue
                f.write(f'{sha256_file(p)}  {os.path.relpath(p, SPRINT)}\n')
    print('data_lineage.json / version_manifest.csv / SHA256SUMS.txt 已写出')
    print('artifacts in lineage:', len(lineage['result_line']) + len(lineage['code_line'])
          + len(lineage['baseline_input_line']) + len(lineage['primary_data_line'])
          + len(lineage['manuscript_line']))


if __name__ == '__main__':
    main()

"""rq1f_patch_timing.py — 给八个基线加**纯计时**插桩（train / predict 耗时）

设计约束（严格遵守"不得改变基线的输入信息、训练质量或正常能力"）：
  * 只插入 time.time() 计时与落盘；**不改动任何超参、损失、优化、选择或数据处理逻辑**；
  * 每处替换都要求"恰好命中一次"，否则直接报错退出（防止误改）；
  * timing.json 写在**与该基线 y_pred.npy 相同的目录**，便于运行器统一收集；
  * 生成的 unified diff 存入 results/rq1f_timing_patch.diff，供审计核对。

输出：每个 <slug>/result[/baseline]/timing.json = {train_seconds, predict_seconds, total_seconds}
"""
import os
import difflib
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BL = os.path.join(HERE, 'code', 'baselines_fixed')

HEADER = ('import time as _rq1f_time\n'
          'def _rq1f_mark():\n'
          '    return _rq1f_time.time()\n')

# (slug, [(anchor, replacement), ...])
PATCHES = {
    '1DCNN-DBO-LSTM': [
        ('import json\n', 'import json\n' + HEADER),
        ('    train_losses, val_losses = model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary)\n',
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    train_losses, val_losses = model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary)\n'
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'),
        ('    y_pred = model.predict(X_test_scaled)\n'
         '    y_pred_proba = model.predict_proba(X_test_scaled)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    y_pred = model.predict(X_test_scaled)\n'
         '    y_pred_proba = model.predict_proba(X_test_scaled)\n'
         '    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n'
         '    json.dump({"train_seconds": round(_rq1f_train_s, 3),\n'
         '               "predict_seconds": round(_rq1f_pred_s, 3),\n'
         '               "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},\n'
         '              open(os.path.join(RESULT_DIR, "timing.json"), "w"))\n'),
    ],
    'AUTDL': [
        ('import json\n', 'import json\n' + HEADER),
        ('    for epoch in range(EPOCHS):\n',
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    for epoch in range(EPOCHS):\n'),
        ('    # Load best model\n',
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'
         '    # Load best model\n'),
        ('    y_pred, y_prob = predict(model, test_loader, device)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    y_pred, y_prob = predict(model, test_loader, device)\n'
         '    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n'
         '    json.dump({"train_seconds": round(_rq1f_train_s, 3),\n'
         '               "predict_seconds": round(_rq1f_pred_s, 3),\n'
         '               "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},\n'
         '              open(os.path.join(RESULT_DIR, "timing.json"), "w"))\n'),
    ],
    'CIC': [
        ('import numpy as np\n', 'import numpy as np\nimport json as _rq1f_json\n' + HEADER),
        ('    model = CICClassifier()\n    model.fit(X_train, y_train, X_val, y_val)\n',
         '    model = CICClassifier()\n'
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    model.fit(X_train, y_train, X_val, y_val)\n'
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'),
        ('    y_pred = model.predict(X_test)\n    y_proba = model.predict_proba(X_test)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    y_pred = model.predict(X_test)\n'
         '    y_proba = model.predict_proba(X_test)\n'
         '    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n'
         '    _rq1f_json.dump({"train_seconds": round(_rq1f_train_s, 3),\n'
         '                     "predict_seconds": round(_rq1f_pred_s, 3),\n'
         '                     "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},\n'
         '                    open(os.path.join(result_dir, "timing.json"), "w"))\n'),
    ],
    'DeepPipe': [
        ('import numpy as np\n', 'import numpy as np\nimport json as _rq1f_json\n' + HEADER),
        ('    model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, epochs=10, batch_size=64)\n',
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, epochs=10, batch_size=64)\n'
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'),
        ('    y_pred, y_pred_proba = model.predict(X_test_scaled)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    y_pred, y_pred_proba = model.predict(X_test_scaled)\n'
         '    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n'
         '    _rq1f_json.dump({"train_seconds": round(_rq1f_train_s, 3),\n'
         '                     "predict_seconds": round(_rq1f_pred_s, 3),\n'
         '                     "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},\n'
         '                    open(os.path.join(result_dir, "timing.json"), "w"))\n'),
    ],
    'DeepPipeNet': [
        ('import numpy as np\n', 'import numpy as np\nimport json as _rq1f_json\n' + HEADER),
        ('    for epoch in range(50):\n',
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    for epoch in range(50):\n'),
        ('    # 8. 评估\n',
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'
         '    # 8. 评估\n'),
        ('    with torch.no_grad():\n        X_test_t = X_test_t.to(device)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    with torch.no_grad():\n        X_test_t = X_test_t.to(device)\n'),
        ("        np.save(os.path.join(RESULT_DIR, 'y_pred.npy'), np.asarray(y_pred))\n",
         "        np.save(os.path.join(RESULT_DIR, 'y_pred.npy'), np.asarray(y_pred))\n"
         "        _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n"
         "        _rq1f_json.dump({'train_seconds': round(_rq1f_train_s, 3),\n"
         "                        'predict_seconds': round(_rq1f_pred_s, 3),\n"
         "                        'total_seconds': round(_rq1f_pred_s + _rq1f_train_s, 3)},\n"
         "                       open(os.path.join(RESULT_DIR, 'timing.json'), 'w'))\n"),
    ],
    'MFF-CNN-LightGBM': [
        ('import numpy as np\n', 'import numpy as np\nimport json as _rq1f_json\n' + HEADER),
        ('    model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, epochs=3, batch_size=512)\n',
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, epochs=3, batch_size=512)\n'
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'),
        ('    y_pred, y_pred_proba = model.predict(X_test_scaled)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    y_pred, y_pred_proba = model.predict(X_test_scaled)\n'
         '    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n'
         '    _rq1f_json.dump({"train_seconds": round(_rq1f_train_s, 3),\n'
         '                     "predict_seconds": round(_rq1f_pred_s, 3),\n'
         '                     "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},\n'
         '                    open(os.path.join(result_dir, "timing.json"), "w"))\n'),
    ],
    'SART-1DCNN': [
        ('import numpy as np\n', 'import numpy as np\nimport json as _rq1f_json\n' + HEADER),
        ('    model = train_model(model, train_loader, val_loader, device, epochs, learning_rate)\n',
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    model = train_model(model, train_loader, val_loader, device, epochs, learning_rate)\n'
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'),
        ('    accuracy, mcc, auc = evaluate_model(model, test_loader, device)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    accuracy, mcc, auc = evaluate_model(model, test_loader, device)\n'
         '    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n'
         '    _rq1f_json.dump({"train_seconds": round(_rq1f_train_s, 3),\n'
         '                     "predict_seconds": round(_rq1f_pred_s, 3),\n'
         '                     "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},\n'
         '                    open(os.path.join(os.path.dirname(__file__), "result", "timing.json"), "w"))\n'),
    ],
    'SMDPSO-SVM': [
        ('import json\n', 'import json\n' + HEADER),
        ('    model.fit(X_train_sub, y_train_sub)\n',
         '    _rq1f_t0 = _rq1f_mark()\n'
         '    model.fit(X_train_sub, y_train_sub)\n'
         '    _rq1f_train_s = _rq1f_mark() - _rq1f_t0\n'),
        ('    y_pred = model.predict(X_test)\n',
         '    _rq1f_t1 = _rq1f_mark()\n'
         '    y_pred = model.predict(X_test)\n'
         '    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1\n'
         '    json.dump({"train_seconds": round(_rq1f_train_s, 3),\n'
         '               "predict_seconds": round(_rq1f_pred_s, 3),\n'
         '               "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},\n'
         '              open(os.path.join(os.path.dirname(__file__), "result", "timing.json"), "w"))\n'),
    ],
}


def main():
    diffs = []
    for slug, reps in PATCHES.items():
        p = os.path.join(BL, slug, 'train.py')
        src = open(p, encoding='utf-8').read()
        orig = src
        for anchor, rep in reps:
            cnt = src.count(anchor)
            if cnt != 1:
                print(f'!! {slug}: 锚点命中 {cnt} 次（应为 1）：{anchor[:70]!r}', file=sys.stderr)
                raise SystemExit(1)
            src = src.replace(anchor, rep)
        open(p, 'w', encoding='utf-8').write(src)
        diffs.extend(difflib.unified_diff(orig.splitlines(True), src.splitlines(True),
                                          fromfile=f'baselines_fixed/{slug}/train.py (原)',
                                          tofile=f'baselines_fixed/{slug}/train.py (加计时)'))
        print(f'[OK] {slug} 已加计时插桩', flush=True)

    out = os.path.join(HERE, 'results', 'rq1f_timing_patch.diff')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(''.join(diffs))
    print(f'[OK] diff -> {out}  ({len(diffs)} 行)')


if __name__ == '__main__':
    main()

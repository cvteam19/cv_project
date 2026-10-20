"""Common metrics for records with original-pixel xyxy boxes; no GPU imports at module level."""
from pathlib import Path
import json
from main_model import iou, operating_metrics


def matched_counts(record, threshold):
    predictions = sorted((p for p in record['pred'] if p[1] >= threshold), key=lambda p: -p[1])
    used, matched = set(), set()
    for k, p in enumerate(predictions):
        overlap, j = max(((iou(p[2:], g[1:]), j) for j, g in enumerate(record['gt'])
                          if j not in used and g[0] == p[0]), default=(0.0, -1))
        if overlap >= 0.5:
            used.add(j)
            matched.add(k)
    return predictions, used, matched


def operating_rows(records, threshold):
    rows = []
    empty = [r for r in records if not r['gt']]
    for cls in (None, 0, 1):
        row = operating_metrics(records, threshold, cls)
        empty_fp = sum(sum(p[1] >= threshold and (cls is None or p[0] == cls)
                           for p in r['pred']) for r in empty)
        row.update(empty_image_fp=empty_fp,
                   fp_per_empty_image=empty_fp/len(empty) if empty else None)
        rows.append(row)
    return rows


def confusion_matrix(records, threshold):
    # Rows=true, columns=predicted: smoke, fire, background.
    # Preserve EXACTLY the class-aware TP matching used for P/R/F1 first.
    matrix = [[0]*3 for _ in range(3)]
    for record in records:
        predictions, used, matched = matched_counts(record, threshold)
        for k in matched:
            cls = predictions[k][0]
            matrix[cls][cls] += 1
        for k, p in enumerate(predictions):
            if k in matched:
                continue
            overlap, j = max(((iou(p[2:], g[1:]), j) for j, g in enumerate(record['gt'])
                              if j not in used and g[0] != p[0]), default=(0.0, -1))
            if overlap >= 0.5:
                used.add(j)
                matrix[record['gt'][j][0]][p[0]] += 1
            else:
                matrix[2][p[0]] += 1
        for j, g in enumerate(record['gt']):
            if j not in used:
                matrix[g[0]][2] += 1
    return dict(labels=['smoke', 'fire', 'background'], rows='true', columns='predicted',
                background_background='not defined for object detection', values=matrix)


def collect_records(predict_fn, root, split, batch_size, out, label):
    import time
    import torch
    from PIL import Image
    from main_model import images, ground_truth
    paths = images(root, split)
    with Image.open(paths[0]) as im:
        warmup = [im.convert('RGB') for _ in range(batch_size)]
    for _ in range(3):
        predict_fn(warmup)
    torch.cuda.synchronize()
    records, seconds = [], 0.0
    with (Path(out)/f'{label}_{split}_predictions.jsonl').open('w', encoding='utf-8') as log:
        for start in range(0, len(paths), batch_size):
            part = paths[start:start+batch_size]
            batch = []
            for path in part:
                with Image.open(path) as im:
                    batch.append(im.convert('RGB'))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = predict_fn(batch)
            torch.cuda.synchronize()
            seconds += time.perf_counter()-t0
            for path, image, output in zip(part, batch, outputs):
                w, h = image.size
                truth = [[g[0], g[1]*w, g[2]*h, g[3]*w, g[4]*h]
                         for g in ground_truth(root, split, path)]
                record = dict(image=str(path), gt=truth, pred=[
                    [int(c), float(s), *box] for c, s, box in zip(
                        output['labels'].detach().cpu().tolist(),
                        output['scores'].detach().cpu().tolist(),
                        output['boxes'].detach().cpu().tolist())])
                records.append(record)
                log.write(json.dumps(record)+'\n')
            if start % 200 == 0:
                print(label, split, min(start+batch_size, len(paths)), '/', len(paths))
    assert len(records) == len(paths)
    return records, dict(batch=batch_size, warmup_batches=3, seconds=seconds,
        ms_per_image=seconds/len(records)*1000, fps=len(records)/seconds,
        scope='RGB PIL -> preprocessing, H2D, inference, postprocessing; excludes decode, D2H outputs, metrics')


def coco_metrics(records):
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True,
                                  max_detection_thresholds=[1, 10, 100])
    for r in records:
        pred = dict(boxes=torch.tensor([p[2:] for p in r['pred']], dtype=torch.float32).reshape(-1, 4),
                    labels=torch.tensor([p[0] for p in r['pred']], dtype=torch.long),
                    scores=torch.tensor([p[1] for p in r['pred']], dtype=torch.float32))
        gt = dict(boxes=torch.tensor([g[1:] for g in r['gt']], dtype=torch.float32).reshape(-1, 4),
                  labels=torch.tensor([g[0] for g in r['gt']], dtype=torch.long))
        metric.update([pred], [gt])
    result = metric.compute()
    return dict(map50=float(result['map_50']), map50_95=float(result['map']),
                per_class_map50_95={['smoke','fire'][int(c)]: float(v)
                    for c, v in zip(result['classes'].tolist(), result['map_per_class'].tolist())})


def save_evaluation(records, threshold, timing, out, name):
    import matplotlib.pyplot as plt
    from main_model import dump, write_csv
    out = Path(out)
    operating = operating_rows(records, threshold)
    matrix = confusion_matrix(records, threshold)
    report = dict(metrics=coco_metrics(records), operating=operating,
                  confusion_matrix=matrix, timing=timing, confidence=threshold,
                  matching_iou=0.5, images=len(records),
                  fp_per_empty_image_target_passed=(operating[0]['fp_per_empty_image'] < 0.5)
                  if operating[0]['fp_per_empty_image'] is not None else None)
    dump(out/f'{name}_metrics.json', report)
    write_csv(out/f'{name}_operating.csv', operating)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix['values'], cmap='Blues')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=matrix['labels'],
           yticklabels=matrix['labels'], xlabel='Predicted', ylabel='True')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix['values'][i][j]) if (i,j)!=(2,2) else 'N/A',
                    ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out/f'{name}_confusion.png', dpi=150)
    plt.close(fig)
    # Candidates for manual failure analysis; do not claim reviewed error categories.
    candidates = []
    for record in records:
        predictions, used, matched = matched_counts(record, threshold)
        fp, fn = len(predictions)-len(matched), len(record['gt'])-len(used)
        if fp or fn:
            candidates.append((fp+fn, fp, fn, record))
    candidates.sort(key=lambda item: -item[0])
    from PIL import Image, ImageDraw
    manifest = []
    for index, (_, fp, fn, record) in enumerate(candidates[:6]):
        with Image.open(record['image']) as im:
            image = im.convert('RGB')
        draw = ImageDraw.Draw(image)
        for gt in record['gt']:
            draw.rectangle(gt[1:], outline='lime', width=2)
        for p in record['pred']:
            if p[1] >= threshold:
                draw.rectangle(p[2:], outline='red', width=2)
                draw.text((p[2], p[3]), f'{["smoke","fire"][p[0]]} {p[1]:.2f}', fill='red')
        filename = f'{name}_error_candidate_{index:02d}.jpg'
        image.save(out/filename)
        manifest.append(dict(image=record['image'], fp=fp, fn=fn, visualization=filename,
                             review_notes=''))
    write_csv(out/f'{name}_error_candidates.csv', manifest,
              ['image', 'fp', 'fn', 'visualization', 'review_notes'])
    return report


def choose_threshold(records, out, name):
    import matplotlib.pyplot as plt
    from main_model import write_csv, dump
    thresholds = [0.001, 0.01, 0.025] + [v/100 for v in range(5, 100, 5)]
    rows = [dict(row) for threshold in thresholds for row in operating_rows(records, threshold)]
    write_csv(Path(out)/f'{name}_validation_threshold_sweep.csv', rows)
    overall = [r for r in rows if r['class_name'] == 'all']
    best = max(overall, key=lambda r: (r['f1'], r['recall'], r['threshold']))
    dump(Path(out)/f'{name}_locked_threshold.json', dict(threshold=best['threshold'],
        split='val', criterion='maximum micro-F1; ties: recall, then higher threshold', matching_iou=0.5))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cls in ['all', 'smoke', 'fire']:
        part = [r for r in rows if r['class_name'] == cls]
        axes[0].plot([r['recall'] for r in part], [r['precision'] for r in part], label=cls)
        axes[1].plot([r['threshold'] for r in part], [r['f1'] for r in part], label=cls)
    axes[0].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel='Confidence threshold', ylabel='F1', ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(out)/f'{name}_validation_sensitivity.png', dpi=150)
    plt.close(fig)
    return best['threshold']

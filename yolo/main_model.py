"""D-Fire main model experiments. Importable helpers; run through the companion notebook."""
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import shutil
import time

NAMES = {0: 'smoke', 1: 'fire'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_label(text):
    """Match Data Lead policy: reject invalid rows, clip valid boxes to [0, 1]."""
    kept, rejected, clipped = [], [], 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(map(math.isfinite, values)):
                raise ValueError('expected five finite numbers')
            c, x, y, w, h = values
            if c not in NAMES or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('invalid class/center/size')
            corners = [x-w/2, y-h/2, x+w/2, y+h/2]
            a, b, d, e = [min(1., max(0., z)) for z in corners]
            if d <= a or e <= b:
                raise ValueError('no visible area')
            clipped += int(corners != [a, b, d, e])
            kept.append(f'{int(c)} {(a+d)/2:.8f} {(b+e)/2:.8f} {d-a:.8f} {e-b:.8f}')
        except ValueError as exc:
            rejected.append({'line': number, 'text': line, 'reason': str(exc)})
    return '\n'.join(kept) + ('\n' if kept else ''), rejected, clipped


def images(root, split):
    return sorted(p for p in (Path(root)/split/'images').rglob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def label_for(root, split, image):
    relative = Path(image).relative_to(Path(root)/split/'images')
    return Path(root)/split/'labels'/relative.with_suffix('.txt')


def locate_prepared(downloaded):
    """Find real image/label folders without trusting stale absolute paths in YAML."""
    downloaded = Path(downloaded)
    candidates = [downloaded] + [p.parent for p in downloaded.rglob('train') if p.is_dir()]
    candidates = sorted(set(p.resolve() for p in candidates if all(
        (p/s/'images').is_dir() and (p/s/'labels').is_dir() for s in ('train', 'val', 'test'))))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one prepared data root, found {candidates}. '
                           'Set SOURCE_DATA to the folder containing train/, val/, test/. '
                           'If the Input contains only an archive, extract it into working first.')
    return candidates[0]


def check_class_mapping(source, expected, search_root=None):
    """Verify published names where available; otherwise record explicit project mapping."""
    source = Path(source).resolve()
    search_root = Path(search_root).resolve() if search_root is not None else source.parent
    configs = sorted(set(search_root.rglob('data*.yaml')) | set(search_root.rglob('data*.yml')))
    verified_files = []
    for path in configs:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(cfg, dict) or 'names' not in cfg:
            continue
        names = cfg['names']
        names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
        if names != expected:
            raise ValueError(f'Unexpected class mapping in {path}: {names}; expected {expected}')
        verified_files.append(str(path))
    if not verified_files:
        print(f'No published YAML with names found. Using explicit project mapping: {expected}. '
              'Label IDs will be validated; their semantic names cannot be inferred from numbers alone.')
    return dict(names=expected, verified_from_yaml=bool(verified_files), yaml_files=verified_files,
                mapping_source='published_yaml' if verified_files else 'explicit_project_configuration')


def prepare(source, dest, expected=None, prepared=False):
    """Writable image COPIES avoid Ultralytics repairing JPEGs in read-only inputs."""
    from PIL import Image
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if dest == source or source in dest.parents or dest in source.parents:
        raise ValueError('Source and prepared output must be separate directory trees')
    if dest.exists():
        raise FileExistsError(f'{dest} already exists. Reuse its config or choose a new output.')
    dest.mkdir(parents=True)
    report, rejects, manifest, hashes = {}, [], [], {}
    for split in ('train', 'val', 'test'):
        files = images(source, split)
        if not files or (expected and len(files) != expected[split]):
            raise ValueError(f'{split}: unexpected image count {len(files)}')
        seen_labels, stats = set(), dict(images=len(files), boxes=0, empty_labels=0, clipped=0, rejected=0)
        for src in files:
            label = label_for(source, split, src)
            if not label.is_file():
                raise FileNotFoundError(f'Missing label is not a verified negative: {label}')
            key = label.relative_to(source/split/'labels')
            if key in seen_labels:
                raise ValueError(f'Ambiguous image-to-label mapping: {key}')
            seen_labels.add(key)
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f'Exact cross-split duplicate: {src}; other split={hashes[digest]}')
            hashes[digest] = split
            rel = src.relative_to(source/split/'images')
            target = dest/split/'images'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            # Verify decode without resizing or EXIF transpose (boxes use stored pixel coordinates).
            with Image.open(src) as im:
                im.load()
            shutil.copy2(src, target)
            original_text = label.read_text(encoding='utf-8')
            text, bad, clipped = clean_label(original_text)
            if prepared:
                original_rows = [list(map(float, line.split())) for line in original_text.splitlines() if line.strip()] if not bad else []
                cleaned_rows = [list(map(float, line.split())) for line in text.splitlines()]
                changed = len(original_rows) != len(cleaned_rows) or any(
                    abs(a-b) > 1e-7 for row_a, row_b in zip(original_rows, cleaned_rows)
                    for a, b in zip(row_a, row_b))
                if bad or changed:
                    raise ValueError(f'Prepared label failed validation: {label}. '
                                     'No automatic annotation changes are allowed in prepared mode.')
                text, clipped = original_text, 0
            out_label = label_for(dest, split, target)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            out_label.write_text(text, encoding='utf-8')
            stats['boxes'] += sum(bool(line.strip()) for line in text.splitlines())
            stats['empty_labels'] += int(not text.strip())
            stats['clipped'] += clipped
            stats['rejected'] += len(bad)
            rejects.extend(dict(split=split, image=str(rel), **row) for row in bad)
            manifest.append(dict(split=split, image=str(rel), sha256=digest))
        report[split] = stats
        print(split, stats)
    # Preserve preparation provenance where supplied by the published dataset.
    for directory in (source.parent, source):
        for name in ('removed_label_rows.csv', 'dataset_summary.json', 'data_lead_final_summary.json'):
            report_path = directory/name
            if report_path.is_file():
                (dest/'source_reports').mkdir(exist_ok=True)
                shutil.copy2(report_path, dest/'source_reports'/name)
    write_csv(dest/'rejected_rows.csv', rejects, ['split', 'image', 'line', 'text', 'reason'])
    write_csv(dest/'manifest.csv', manifest)
    dump(dest/'preparation.json', dict(splits=report, prepared_input=prepared,
         source=str(source), labels_modified=not prepared))  # Completion marker written last.
    return report


def make_yaml(path, root, train=None):
    import yaml
    config = dict(path=str(Path(root).resolve()), train=train or 'train/images',
                  val='val/images', test='test/images', names=NAMES, nc=2)
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return str(Path(path).resolve())


def load_model(weights, architecture='yolo'):
    from ultralytics import YOLO, RTDETR
    return (RTDETR if architecture == 'rtdetr' else YOLO)(str(weights))


def train(weights, data, out, name, epochs=50, batch=8, lr=0.001,
          architecture='yolo', patience=8, resume=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU accelerator before training.')
    run = Path(out)/name
    if resume:
        model = load_model(resume, architecture)
        model.train(resume=True, device=0)
    else:
        if run.exists():
            raise FileExistsError(f'{run} exists. Use resume=last.pt or a new name.')
        model = load_model(weights, architecture)
        start = time.time()
        model.train(data=str(data), epochs=epochs, imgsz=640, batch=batch,
                    device=0, workers=2, optimizer='AdamW', lr0=lr, lrf=0.01,
                    weight_decay=0.0005, warmup_epochs=3, cos_lr=True,
                    seed=42, deterministic=True, amp=True, cache=False,
                    patience=patience, close_mosaic=min(5, epochs),
                    mosaic=1.0, mixup=0.0, fliplr=0.5, flipud=0.0,
                    save=True, save_period=5, plots=True,
                    project=str(out), name=name, exist_ok=False)
        dump(run/'elapsed.json', {'train_seconds': time.time()-start})
    save_dir = Path(model.trainer.save_dir)
    best = save_dir/'weights'/'best.pt'
    if not best.is_file():
        raise FileNotFoundError(best)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return str(best)


def evaluate(weights, data, out, name, size=640, split='val', architecture='yolo'):
    model = load_model(weights, architecture)
    metrics = model.val(data=str(data), split=split, imgsz=size, batch=4,
                        device=0, workers=2, conf=0.001, iou=0.7, max_det=300,
                        rect=False, plots=True, project=str(out), name=name,
                        exist_ok=False)
    row = dict(name=name, checkpoint=str(weights), architecture=architecture,
               split=split, imgsz=size, map50=float(metrics.box.map50),
               map50_95=float(metrics.box.map),
               inference_ms_per_image=float(metrics.speed['inference']))
    # AP class indices are explicit; absent classes must not inherit the mean AP.
    for cls, ap in zip(metrics.box.ap_class_index, metrics.box.ap):
        row[f'{NAMES[int(cls)]}_map50_95'] = float(ap)
    dump(Path(metrics.save_dir)/'summary.json', row)
    return row


def ground_truth(root, split, image):
    result = []
    for line in label_for(root, split, image).read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        result.append([int(c), x-w/2, y-h/2, x+w/2, y+h/2])
    return result


def predict_records(weights, root, split, size=640, architecture='yolo', subset=None):
    model = load_model(weights, architecture)
    paths = subset if subset is not None else images(root, split)
    records = []
    # Bound the source list per call; stream results, retaining only boxes, not image arrays.
    for start in range(0, len(paths), 128):
        results = model.predict(source=[str(p) for p in paths[start:start+128]],
                                stream=True, imgsz=size, batch=4, device=0,
                                conf=0.001, iou=0.7, max_det=300, rect=False, verbose=False)
        for result in results:
            boxes = result.boxes
            predictions = [[int(c), float(s), *map(float, b)] for c, s, b in
                           zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                               boxes.xyxyn.cpu().tolist())]
            records.append(dict(image=str(result.path),
                                gt=ground_truth(root, split, result.path), pred=predictions))
    if len(records) != len(paths):
        raise RuntimeError('Prediction count mismatch')
    return records


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = lambda box: max(0., box[2]-box[0])*max(0., box[3]-box[1])
    union = area(a)+area(b)-intersection
    return intersection/union if union else 0.


def operating_metrics(records, threshold, cls=None):
    tp = fp = fn = negative_count = false_alarms = 0
    for record in records:
        gt = [g for g in record['gt'] if cls is None or g[0] == cls]
        predictions = sorted([p for p in record['pred'] if p[1] >= threshold and
                              (cls is None or p[0] == cls)], key=lambda p: -p[1])
        used = set()
        for p in predictions:
            candidates = [(iou(p[2:], g[1:]), j) for j, g in enumerate(gt)
                          if j not in used and g[0] == p[0]]
            overlap, j = max(candidates, default=(0., -1))
            if overlap >= 0.5:
                tp += 1
                used.add(j)
            else:
                fp += 1
        fn += len(gt)-len(used)
        # Negative means no annotations of either class, including for per-class FAR.
        if not record['gt']:
            negative_count += 1
            false_alarms += int(bool(predictions))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(threshold=threshold, class_name=NAMES.get(cls, 'all'), tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                negative_images=negative_count, false_alarm_images=false_alarms,
                false_alarm_rate=false_alarms/negative_count if negative_count else None)


def mine(weights, root, out, limit=200):
    """Rank prepared empty-label train images; exclude known rejected-label cases from audit reports."""
    source = Path(root)
    negatives = [p for p in images(source, 'train') if not label_for(source, 'train', p).read_text().strip()]
    # preparation's rejected_rows prevents mislabeled positives from becoming hard negatives.
    rejected = set()
    for report_path in (source/'rejected_rows.csv', source/'source_reports'/'removed_label_rows.csv'):
        if report_path.is_file():
            with report_path.open(encoding='utf-8') as f:
                rejected.update(r['image'].replace('\\', '/') for r in csv.DictReader(f)
                                if r.get('split') == 'train' and r.get('image'))
    negatives = [p for p in negatives if p.relative_to(source/'train'/'images').as_posix() not in rejected]
    records = predict_records(weights, root, 'train', subset=negatives)
    ranked = sorted(records, key=lambda r: max([p[1] for p in r['pred']], default=0.), reverse=True)
    rows = [dict(image=r['image'], max_conf=max(p[1] for p in r['pred']),
                 approved='', category='', notes='') for r in ranked if r['pred']][:limit]
    write_csv(Path(out)/'hard_negative_review.csv', rows,
              ['image', 'max_conf', 'approved', 'category', 'notes'])
    # Paginated contact sheets; inspect full images separately when uncertain.
    import matplotlib.pyplot as plt
    from PIL import Image
    for start in range(0, len(rows), 20):
        fig, axes = plt.subplots(4, 5, figsize=(18, 12))
        for ax, row in zip(axes.flat, rows[start:start+20]):
            with Image.open(row['image']) as im:
                ax.imshow(im.convert('RGB'))
            ax.set_title(f"{Path(row['image']).name}\nscore={row['max_conf']:.3f}", fontsize=8)
        for ax in axes.flat:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(Path(out)/f'hard_negative_review_{start//20:02d}.jpg', dpi=100)
        plt.close(fig)
    return rows


def ablation_lists(root, review, out, repeats=2):
    """Equal length = equal optimizer steps; base data retained in both arms."""
    base = [str(p.resolve()) for p in images(root, 'train')]
    membership = set(base)
    with Path(review).open(encoding='utf-8') as f:
        reviewed = list(csv.DictReader(f))
    approved = sorted({str(Path(r['image']).resolve()) for r in reviewed
                       if r['approved'].strip().lower() == 'yes'})
    if not approved:
        raise ValueError('Review candidates first; set approved=yes only for confirmed backgrounds.')
    for p in approved:
        if p not in membership or label_for(root, 'train', p).read_text().strip():
            raise ValueError(f'Approved image is not an empty-label TRAIN image: {p}')
    negatives = [p for p in base if not label_for(root, 'train', p).read_text().strip()]
    count = len(approved)*repeats
    rng = random.Random(42)
    # Controls repeat randomly sampled negatives; treatment repeats reviewed hard negatives.
    control = base + rng.choices(negatives, k=count)
    treatment = base + approved*repeats
    assert len(control) == len(treatment)
    paths = []
    for name, entries in [('control', control), ('hard_negative', treatment)]:
        listing = Path(out)/f'{name}_train.txt'
        listing.write_text('\n'.join(entries)+'\n', encoding='utf-8')
        paths.append(make_yaml(Path(out)/f'{name}.yaml', root, str(listing.resolve())))
    dump(Path(out)/'ablation_design.json', dict(base_images=len(base), extra_samples=count,
         images_per_epoch=len(control), approved=approved, seed=42,
         policy='same checkpoint, epochs, optimizer, batch; random negatives vs reviewed hard negatives'))
    return paths

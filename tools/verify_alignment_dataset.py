from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import validate_roi

EXT = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}


def images(root: Path):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXT)


def scenario(rel: Path) -> str:
    p = rel.parts
    if p and p[0].lower() == 'good': return 'good'
    if len(p) >= 2 and p[0].lower() == 'ng': return p[1]
    return p[0] if p else 'unknown'


def check_config(ref, cfg):
    h,w = ref.shape[:2]
    cw,ch = cfg.get('reference_width'), cfg.get('reference_height')
    if cw is not None and ch is not None and (int(cw),int(ch)) != (w,h):
        raise SystemExit(f'CONFIG/REFERENCE SIZE MISMATCH: config={cw}x{ch}, reference={w}x{h}. Do not auto-remap ROIs.')
    for x in cfg.get('screw_slots',[]):
        if x.get('enabled',True) and not validate_roi(x.get('roi'),w,h):
            raise SystemExit(f"Invalid ROI {x.get('id')}: {x.get('roi')}")


def draw(img,cfg):
    for x in cfg.get('screw_slots',[]):
        if not x.get('enabled',True): continue
        roi=x.get('roi')
        if roi is None: continue
        a,b,c,d=map(int,roi)
        exp=str(x.get('expected',''))
        color=(0,190,0) if exp=='screw' else (0,150,220)
        cv2.rectangle(img,(a,b),(a+c,b+d),color,2)
        cv2.putText(img,f"{x.get('id')}:{exp}",(a,max(20,b-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser(description='Verify alignment only. No screw detection or anomaly scoring.')
    p.add_argument('--input-root',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--scenario',action='append')
    p.add_argument('--foreground-threshold',type=int,default=238)
    p.add_argument('--allow-foreground-fallback',action='store_true')
    a=p.parse_args()

    root=Path(a.input_root).resolve(); out=Path(a.output).resolve()
    ref=read_image(a.reference); cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    check_config(ref,cfg)
    filters={str(x).lower() for x in (a.scenario or [])}
    selected=[]
    for f in images(root):
        rel=f.relative_to(root); sc=scenario(rel)
        if not filters or sc.lower() in filters: selected.append((f,rel,sc))

    acfg=ProductLocatorConfig(foreground_threshold=a.foreground_threshold)
    rows=[]
    print('=== Alignment-only verification ===')
    print(f'Images: {len(selected)} | detection: OFF')
    for i,(f,rel,sc) in enumerate(selected,1):
        try:
            r=align_to_reference(read_image(f),ref,acfg)
            status='ALIGN_OK' if (r.feature_matrix is not None or a.allow_foreground_fallback) else 'RETRY'
            od=out/'overlays'/rel.with_suffix('.jpg'); ad=out/'aligned'/rel.with_suffix('.png')
            od.parent.mkdir(parents=True,exist_ok=True); ad.parent.mkdir(parents=True,exist_ok=True)
            write_image(ad,r.aligned); ov=r.aligned.copy(); draw(ov,cfg); write_image(od,ov)
            rows.append({'path':rel.as_posix(),'scenario':sc,'status':status,'method':r.method,'inlier_ratio':r.feature_inlier_ratio,'ecc':r.ecc_score,'overlay':str(od),'error':''})
            print(f'[{i}/{len(selected)}] {rel.as_posix()} -> {status} | {r.method} | ecc={r.ecc_score}')
        except Exception as e:
            rows.append({'path':rel.as_posix(),'scenario':sc,'status':'RETRY','method':'','inlier_ratio':'','ecc':'','overlay':'','error':str(e)})
            print(f'[{i}/{len(selected)}] {rel.as_posix()} -> RETRY: {e}')

    out.mkdir(parents=True,exist_ok=True)
    with (out/'alignment_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else ['path','scenario','status','method','inlier_ratio','ecc','overlay','error']); w.writeheader(); w.writerows(rows)
    ok=sum(r['status']=='ALIGN_OK' for r in rows)
    (out/'summary.json').write_text(json.dumps({'images':len(rows),'align_ok':ok,'retry':len(rows)-ok},indent=2),encoding='utf-8')
    print(f'ALIGN_OK / RETRY: {ok} / {len(rows)-ok}')

if __name__=='__main__': main()

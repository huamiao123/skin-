"""Verify original IMA files without model/test performance access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

from .metadata_protocol import FIELDS, read_csv, write_csv, file_sha, write_json

AUDIT_FIELDS = FIELDS + ['audit_complete', 'image_sha256', 'mask_sha256', 'actual_mask_md5',
    'image_pixel_sha256', 'original_h', 'original_w', 'mask_h', 'mask_w', 'image_mode', 'mask_mode',
    'mask_value_min', 'mask_value_max', 'mask_values', 'spatial_size_aligned', 'audit_error']


def hashes(path):
    sha = hashlib.sha256(); md5 = hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): sha.update(b); md5.update(b)
    return sha.hexdigest(), md5.hexdigest()


def dhash(image):
    gray = np.asarray(image.convert('L').resize((9,8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits((gray[:,1:]>gray[:,:-1]).reshape(-1)).tobytes(), 'big')


def inspect_image(row):
    path = Path(row['image_path'])
    result = {'image_id':row['image_id'], 'image_path':str(path), 'error':''}
    try:
        sha, md5 = hashes(path)
        with Image.open(path) as im:
            im.load(); rgb=im.convert('RGB')
            result.update(image_sha256=sha, image_md5=md5,
                image_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), original_w=im.width,
                original_h=im.height, image_mode=im.mode, dhash=f'{dhash(rgb):016x}')
            expected=(int(row['metadata_width']),int(row['metadata_height']))
            if im.size != expected: result['metadata_size_difference']=f'{expected} vs {im.size}'
    except Exception as exc: result['error']=f'{type(exc).__name__}: {exc}'
    return result


def inspect_mask(row):
    result={'reference_id':row['reference_id'], 'mask_path':row['mask_path'], 'error':''}
    try:
        sha, md5 = hashes(Path(row['mask_path']))
        with Image.open(row['mask_path']) as mask:
            mask.load()
            if mask.mode not in {'1','L','P'}: raise ValueError(f'unexpected mask mode: {mask.mode}')
            values=np.flatnonzero(mask.convert('L').histogram())
            result.update(mask_sha256=sha, actual_mask_md5=md5, mask_h=mask.height,
                mask_w=mask.width, mask_mode=mask.mode, mask_value_min=int(values.min()),
                mask_value_max=int(values.max()), mask_values=';'.join(map(str,values.tolist())))
            observed=set(values.tolist())
            if not (observed <= {0,1} or observed <= {0,255}): raise ValueError('mask contains nonbinary values')
        if md5 != row['mask_md5']: raise ValueError('original mask MD5 does not match official CSV')
    except Exception as exc: result['error']=f'{type(exc).__name__}: {exc}'
    return result


def duplicate_groups(images, field):
    groups=defaultdict(list)
    for image_id, info in images.items():
        if info.get(field):groups[info[field]].append(image_id)
    return [sorted(ids) for ids in groups.values() if len(ids)>1]


def alignment_previews(rows, output, count=16):
    """Fixed evenly spaced RGB/reference overlays for file alignment inspection."""
    paths=[];colors=[(0,255,0),(255,80,0),(0,160,255),(240,0,255),(255,255,0)]
    for split in ['train','val','test']:
        grouped=defaultdict(list)
        for row in rows:
            if row['subset']=='M' and row['split']==split:grouped[row['image_id']].append(row)
        ids=sorted(grouped)
        if not ids:continue
        selected=[ids[i] for i in sorted(set(np.linspace(0,len(ids)-1,min(count,len(ids))).astype(int).tolist()))]
        canvas=Image.new('RGB',(1280,270*((len(selected)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
        for n,image_id in enumerate(selected):
            refs=grouped[image_id];im=Image.open(refs[0]['image_path']).convert('RGB');im.thumbnail((320,240));array=np.array(im)
            for k,ref in enumerate(refs):
                with Image.open(ref['mask_path']) as mask:m=np.array(mask.convert('L').resize(im.size,Image.Resampling.NEAREST))>0
                edge=np.zeros_like(m);edge[1:]|=m[1:]!=m[:-1];edge[:,1:]|=m[:,1:]!=m[:,:-1]
                array[edge]=colors[k%len(colors)]
            x=(n%4)*320;y=(n//4)*270;canvas.paste(Image.fromarray(array),(x,y));draw.text((x+5,y+242),image_id+f' r={len(refs)}',fill='black')
        target=output/f'alignment_fixed16_{split}.jpg';canvas.save(target,quality=92)
        paths.append({'split':split,'artifact':str(target),'image_ids':selected,'purpose':'original file spatial alignment only; no model prediction'})
    write_json(output/'alignment_preview_index.json',paths)
    return paths


def audit(manifest: Path, output: Path, *, split=None, limit=None, workers=4):
    rows=read_csv(manifest)
    if split: rows=[r for r in rows if r['split']==split]
    if limit:
        # A fixed subset of available original training files, used only for T0.
        candidates=sorted({r['image_id'] for r in rows if Path(r['image_path']).is_file()})
        ids=set(candidates[:limit]); rows=[r for r in rows if r['image_id'] in ids and r['subset']=='M']
    output.mkdir(parents=True,exist_ok=True)
    unique_images={r['image_id']:r for r in rows}
    unique_masks={r['reference_id']:r for r in rows}
    with ThreadPoolExecutor(workers) as pool:
        images={x['image_id']:x for x in pool.map(inspect_image,unique_images.values())}
        masks={x['reference_id']:x for x in pool.map(inspect_mask,unique_masks.values())}
    final=[]; errors=[]
    for row in rows:
        image=images[row['image_id']];mask=masks[row['reference_id']]
        error=' | '.join(e for e in [image['error'],mask['error']] if e)
        aligned=not error and (image['original_w'],image['original_h'])==(mask['mask_w'],mask['mask_h'])
        if not error and not aligned:error='RGB/mask original dimensions differ'
        result={**row,**{k:image[k] for k in AUDIT_FIELDS if k in image},
                **{k:mask[k] for k in AUDIT_FIELDS if k in mask},
                'audit_complete':str(not error).lower(),'spatial_size_aligned':str(bool(aligned)).lower(),
                'audit_error':error}
        final.append(result)
        if error:errors.append({'image_id':row['image_id'],'reference_id':row['reference_id'],'error':error})
    exact=duplicate_groups(images,'image_sha256');pixel=duplicate_groups(images,'image_pixel_sha256')
    near=[]
    ids=sorted(i for i in images if not images[i]['error'])
    for n,a in enumerate(ids):
        ah=int(images[a]['dhash'],16)
        for b in ids[n+1:]:
            distance=(ah^int(images[b]['dhash'],16)).bit_count()
            if distance<=4:near.append({'image_a':a,'image_b':b,'dhash_distance':distance,
                'split_a':unique_images[a]['split'],'split_b':unique_images[b]['split'],
                'group_a':unique_images[a]['group_id'],'group_b':unique_images[b]['group_id']})
    cross_exact=[pair for group in exact+pixel for pair in [(a,b) for n,a in enumerate(group) for b in group[n+1:]]
                 if unique_images[pair[0]]['split']!=unique_images[pair[1]]['split']]
    group_splits=defaultdict(set)
    for row in rows:group_splits[row['group_id']].add(row['split'])
    cross_groups={g:sorted(s) for g,s in group_splits.items() if len(s)>1}
    near_cross=[r for r in near if r['split_a']!=r['split_b'] and r['group_a']!=r['group_b']]
    review_path=output/'near_duplicate_review.csv'
    reviews=read_csv(review_path) if review_path.is_file() else []
    reviewed_distinct=set()
    reviewed_linked=set()
    for review in reviews:
        a,b=review['image_a'],review['image_b']
        if (a in images and b in images
                and review.get('image_a_sha256')==images[a].get('image_sha256')
                and review.get('image_b_sha256')==images[b].get('image_sha256')):
            if review.get('decision')=='distinct_images':reviewed_distinct.add((a,b))
            elif review.get('decision') in {'confirmed_same_image','uncertain'} and unique_images[a]['group_id']==unique_images[b]['group_id']:
                reviewed_linked.add((a,b))
    unresolved_near=[r for r in near if (r['image_a'],r['image_b']) not in reviewed_distinct|reviewed_linked]
    # Candidate pairs require visual review; detection is never automatic confirmation.
    scope='T0_training_subset' if limit else (f'{split}_only' if split else 'all_selected_M_H_T1')
    archive=Path(next(iter(unique_images.values()))['image_path']).parents[1]/'downloads/segs.zip' if unique_images else None
    archive_md5=hashes(archive)[1] if archive and archive.is_file() else None
    archive_verified=archive_md5=='c5143aed98283ab57de2c86ce50973c8'
    complete=bool(rows) and not errors and not cross_exact and not cross_groups and not unresolved_near and (bool(limit) or archive_verified)
    target=output/'final_references.csv';write_csv(target,final,AUDIT_FIELDS)
    write_csv(output/'near_duplicate_candidates.csv',near,
              ['image_a','image_b','dhash_distance','split_a','split_b','group_a','group_b'])
    report={'protocol_id':'P2-IMA-M-v2','scope':scope,'file_audit_complete':complete,
            'input_manifest_sha256':file_sha(manifest),'final_references_sha256':file_sha(target),
            'unique_images_requested':len(unique_images),'unique_references_requested':len(unique_masks),
            'decoded_image_count':sum(not v['error'] for v in images.values()),
            'valid_md5_binary_mask_count':sum(not v['error'] for v in masks.values()),
            'validated_reference_rows':sum(r['audit_complete']=='true' for r in final),
            'image_counts_by_subset_split':{s:dict(Counter(r['split'] for r in {x['image_id']:x for x in final if x['subset']==s and x['audit_complete']=='true'}.values())) for s in ['M','H','T1']},
            'file_errors':errors,'exact_file_duplicate_groups':exact,'exact_pixel_duplicate_groups':pixel,
            'cross_split_exact_duplicate_pairs':cross_exact,'known_group_cross_split':cross_groups,
            'near_duplicate_detector':'64-bit grayscale difference hash, Hamming <=4; candidates require review',
            'near_duplicate_candidate_count':len(near),'cross_split_near_duplicate_candidates':near_cross,
            'near_duplicate_review_sha256':file_sha(review_path) if review_path.is_file() else None,
            'near_duplicate_distinct_pairs_reviewed':len(reviewed_distinct),'near_duplicate_unresolved_candidates':unresolved_near,
            'near_duplicate_confirmed_or_conservative_linked_pairs_reviewed':len(reviewed_linked),
            'original_segmentation_archive_md5':archive_md5,'original_segmentation_archive_verified':archive_verified,
            'spatial_alignment_check':'exact original image/mask dimensions; visual overlays additionally required',
            'metadata_dimension_differences':[v for v in images.values() if v.get('metadata_size_difference')],
            'models_evaluated_on_test':False}
    write_json(output/'file_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'file_errors','metadata_dimension_differences','cross_split_near_duplicate_candidates','near_duplicate_unresolved_candidates'}},indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--split',choices=['train','val','test'])
    p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--wait',action='store_true',help='Wait for all selected RGBs and verified archive to exist before full audit')
    a=p.parse_args()
    config={}
    if a.config:
        import yaml
        config=yaml.safe_load(a.config.read_text()) or {}
    a.manifest=a.manifest or Path(config.get('audit_input_manifest',config.get('manifest','data_manifests/selected_references.csv')))
    a.output=a.output or (Path(config['audit']).parent if config.get('audit') else Path('data_manifests'))
    if a.wait:
        pending_rows=read_csv(a.manifest);wanted={r['image_path'] for r in pending_rows}
        archive=Path(pending_rows[0]['image_path']).parents[1]/'downloads/segs.zip'
        while True:
            available=sum(Path(p).is_file() for p in wanted)
            print(f'Waiting for originals: RGB {available}/{len(wanted)}, archive_complete={archive.is_file()}',flush=True)
            if available==len(wanted) and archive.is_file():break
            time.sleep(30)
    audit(a.manifest,a.output,split=a.split,limit=a.limit,workers=a.workers)


if __name__=='__main__':main()

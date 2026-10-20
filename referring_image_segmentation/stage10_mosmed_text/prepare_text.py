"""Encode image descriptions and one constant prompt with the verified frozen CLIP."""
from common import *
import numpy as np, torch

def main():
    from transformers import CLIPTextModel, CLIPTokenizer
    folder=PROJECT/'stage8_text_baselines/assets/clip-vit-base-patch32'
    old=json.loads((PROJECT/'stage8_text_baselines/text_provenance.json').read_text()); assert old['passed']
    for name,digest in old['files'].items(): assert sha(folder/name)==digest,name
    tokenizer=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
    model=CLIPTextModel.from_pretrained(folder,local_files_only=True).eval().cuda().requires_grad_(False)
    rows=json.loads((ASSETS/'all_rows.json').read_text()); texts=[r['text'] for r in rows]+[FIXED_TEXT]
    lengths=[len(t) for t in tokenizer(texts,truncation=False)['input_ids']]
    assert max(lengths)<=77, max(lengths)
    torch.set_num_threads(4); features=[]
    with torch.no_grad():
        for start in range(0,len(texts),256):
            tokens=tokenizer(texts[start:start+256],padding='max_length',max_length=77,truncation=False,return_tensors='pt')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x=model(**{k:v.cuda() for k,v in tokens.items()}).pooler_output
            assert torch.isfinite(x).all(); features.append(x.float().cpu().numpy())
    np.save(ASSETS/'text512.npy',np.concatenate(features))
    dump(ROOT/'text_provenance.json',dict(passed=True,source=str(folder),frozen=True,
        source_revision=old['revision'],source_file_hashes=old['files'],parameters=sum(p.numel() for p in model.parameters()),
        fixed_text=FIXED_TEXT,fixed_index=len(rows),encoding='512-dimensional CLIP text EOS hidden state, then trainable 512->64 projection',
        truncated=0,max_token_length=max(lengths),unique_texts=len(set(texts)),
        row_ids=[r['id'] for r in rows],text_sha256=sha(ASSETS/'text512.npy'),text_input_sha256=sha(ASSETS/'all_rows.json')))
    print('TEXT_READY',len(rows),len(set(texts)),'max_tokens',max(lengths),flush=True)

if __name__=='__main__':main()

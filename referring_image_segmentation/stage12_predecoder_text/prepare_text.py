"""Cache frozen CLIP word-level features, excluding BOS/EOS/PAD from attention."""
from common import *
import numpy as np,torch

def main():
    from transformers import CLIPTextModel,CLIPTokenizer
    folder=PROJECT/'stage8_text_baselines/assets/clip-vit-base-patch32'
    source=json.loads((PROJECT/'stage8_text_baselines/text_provenance.json').read_text());assert source['passed']
    for name,digest in source['files'].items():assert sha(folder/name)==digest,name
    tok=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
    model=CLIPTextModel.from_pretrained(folder,local_files_only=True).cuda().eval().requires_grad_(False)
    rows=json.loads((ASSETS/'all_rows.json').read_text());texts=sorted({r['text'] for r in rows}|{FIXED_TEXT})
    lengths=[len(t) for t in tok(texts,truncation=False)['input_ids']];assert max(lengths)<=77
    tokens=tok(texts,padding='max_length',max_length=77,truncation=False,return_tensors='pt')
    valid=tokens['attention_mask'].bool() & (tokens['input_ids']!=tok.bos_token_id) & (tokens['input_ids']!=tok.eos_token_id)
    assert valid.any(1).all()
    torch.set_num_threads(4)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(**{k:v.cuda() for k,v in tokens.items()})
    words=output.last_hidden_state.float().cpu().numpy();assert np.isfinite(words).all()
    mapping={s:i for i,s in enumerate(texts)};indices=np.array([mapping[r['text']] for r in rows],np.int64)
    old=np.load(ASSETS/'text512.npy');pool=output.pooler_output.float().cpu().numpy()
    errors=np.abs(pool[indices]-old[:-1]);fixed_error=np.abs(pool[mapping[FIXED_TEXT]]-old[-1])
    assert float(errors.max())<.1 and float(fixed_error.max())<.1
    np.save(TOKEN_ASSETS/'word_features512.npy',words)
    np.save(TOKEN_ASSETS/'word_valid.npy',valid.numpy())
    np.save(TOKEN_ASSETS/'row_word_index.npy',indices)
    dump(TOKEN_ASSETS/'text_vocabulary.json',dict(texts=texts,fixed_index=mapping[FIXED_TEXT],row_ids=[r['id'] for r in rows]))
    dump(ROOT/'token_provenance.json',dict(passed=True,source=str(folder),source_revision=source['revision'],source_file_hashes=source['files'],
        frozen=True,parameters=sum(p.numel() for p in model.parameters()),unique_texts=len(texts),max_token_length=max(lengths),truncated=0,
        token_features='CLIP last_hidden_state 77x512; contextual word tokens; no CLIP image encoder',
        attention_mask='Only real non-BOS/non-EOS tokens; PAD and special tokens ignored for cross-attention',
        sentence_cache_check_max_absolute_error=float(errors.max()),fixed_sentence_cache_max_absolute_error=float(fixed_error.max()),
        historical_texts_sha256=sha(ASSETS/'all_rows.json'),files={str(p):sha(p) for p in TOKEN_ASSETS.iterdir() if p.is_file()}))
    print('WORD_TOKENS_READY',len(rows),len(texts),'max_length',max(lengths),'pool_check_error',float(errors.max()),flush=True)

if __name__=='__main__':main()

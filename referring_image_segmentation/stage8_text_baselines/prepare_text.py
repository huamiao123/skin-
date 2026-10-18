"""Pinned CLIP text weights and sentence caches shared by every visual branch."""
from common import *
import numpy as np,torch
from prepare_data import download

REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
FILES=('config.json','merges.txt','vocab.json','tokenizer_config.json','special_tokens_map.json','pytorch_model.bin')

def main():
    folder=ASSETS/'clip-vit-base-patch32';folder.mkdir(exist_ok=True,parents=True)
    for f in FILES:
        download(f'https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/{REV}/{f}',folder/f)
        print('CLIP_FILE',f,flush=True)
    from transformers import CLIPTextModel,CLIPTokenizer
    tokenizer=CLIPTokenizer.from_pretrained(folder,local_files_only=True)
    model,loading=CLIPTextModel.from_pretrained(folder,local_files_only=True,output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs'),loading
    model=model.eval().cuda().requires_grad_(False)
    assert model.config.hidden_size==512 and model.config.max_position_embeddings==77
    torch.set_num_threads(4)
    rows=[r for split in ('train','val','testA','testB') for r in json.loads((ASSETS/f'{split}.json').read_text())]
    texts=[r['expression'] for r in rows]+['segment the object']
    output=np.lib.format.open_memmap(ASSETS/'text512.npy',mode='w+',dtype='float32',shape=(len(texts),512))
    truncated=0
    with torch.no_grad():
        for start in range(0,len(texts),256):
            batch=texts[start:start+256]
            raw=tokenizer(batch,add_special_tokens=True,truncation=False)['input_ids'];truncated+=sum(len(x)>77 for x in raw)
            tokens=tokenizer(batch,padding='max_length',truncation=True,max_length=77,return_tensors='pt')
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**{k:v.cuda() for k,v in tokens.items()}).pooler_output
            assert torch.isfinite(out).all();output[start:start+len(batch)]=out.float().cpu().numpy()
    output.flush();del output
    dump(ASSETS/'text_index.json',dict(sample_ids=[r['sample_id'] for r in rows],fixed_index=len(rows)))
    dump(ROOT/'text_provenance.json',dict(passed=True,model='openai/clip-vit-base-patch32',revision=REV,
        files={f:sha(folder/f) for f in FILES},download_endpoint='hf-mirror.com (official HF endpoint returned429)',hidden_size=512,pooling='CLIPTextModel.pooler_output at EOS, before CLIP visual/text projection',
        frozen=True,max_tokens=77,truncated_expressions=truncated,cache_sha256=sha(ASSETS/'text512.npy'),
        fixed_text='segment the object',loading_info=loading,torch_version=torch.__version__))
    print('TEXT READY',len(rows),truncated,flush=True)

if __name__=='__main__':main()

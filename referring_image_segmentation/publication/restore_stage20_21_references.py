"""Fetch pinned upstream research sources and verify their exact SHA256.

The public archive contains our adaptation code and result artifacts but does
not redistribute these third-party files. Review the upstream licenses before
using them. This script never overwrites an existing file with different bytes.
"""
import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SOURCES = [
    ('Momentisbeused/HEDS-Net', 'a9287da787957b6376a9c403a56ad6693cd04191',
     'models/axis_bridge.py', '5a054761661647b0ed8fe8d2bf67f4878c0f9aafdaf75d2e6f68e482a64bddf1',
     ['stage20_decoder_fusion_seed42/references/axis_bridge_source.py']),
    ('yunxiangfu2001/SegMAN', '9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8',
     'segmentation/mmseg/models/decode_heads/segman_decoder.py',
     '67bea12cb22409344e59908c2ee4b916adcad76be6fb7b2ea43fce988e5b1516',
     ['stage20_decoder_fusion_seed42/references/segman_decoder_source.py']),
    ('yunxiangfu2001/SegMAN', '9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8',
     'segmentation/mmseg/models/decode_heads/csm_triton.py',
     '533c1eaf78853dc8d6e03eea6682d75a15836bb9da2bf290286e0a784df828ad',
     ['stage20_decoder_fusion_seed42/references/segman_csm_triton_source.py']),
    ('YueyueZhu/HGM', '49ae05afc9785dbbc5a303d39e63588b068e5328',
     'lib/Decoder.py', '8c783b14dcedd13d98c433213d2a8f33c8ea0aec5bb4fa8281914b2d73c7a8da',
     ['stage20_decoder_fusion_seed42/references/hgm_decoder_source.py',
      'stage21_hgm_bmd_seed42/references/hgm_decoder_source.py']),
    ('YueyueZhu/HGM', '49ae05afc9785dbbc5a303d39e63588b068e5328',
     'lib/FSEM.py', '107b4fd749bf6443b1e55e603216bca43c2e97d9878d747a392d260e10b6367b',
     ['stage20_decoder_fusion_seed42/references/hgm_fsem_source.py',
      'stage21_hgm_bmd_seed42/references/hgm_fsem_source.py']),
    ('YueyueZhu/HGM', '49ae05afc9785dbbc5a303d39e63588b068e5328',
     'lib/PSA.py', '367f99050964ef7ab15a2291dd9304d1330da56fc47d846ca2a7c6883cea8c0e',
     ['stage21_hgm_bmd_seed42/references/hgm_psa_source.py']),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=PROJECT,
                        help='project directory for restored references')
    args = parser.parse_args()
    for repository, revision, source, expected, destinations in SOURCES:
        url = f'https://raw.githubusercontent.com/{repository}/{revision}/{source}'
        request = urllib.request.Request(url, headers={'User-Agent':'RIS-research-archive'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(1_000_001)
        if len(data)>1_000_000 or sha(data)!=expected:
            raise RuntimeError(f'Upstream content failed SHA256 verification: {url}')
        for relative in destinations:
            path=args.destination/relative
            if path.exists():
                if sha(path.read_bytes()) != expected:
                    raise FileExistsError(f'Existing file has unexpected contents: {path}')
                print(f'verified {path}')
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                temporary=Path(f.name)
                f.write(data)
            temporary.replace(path)
            print(f'restored {path}')


if __name__ == '__main__':
    main()

set -eu
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd /home/featurize/work/mambavision_seg_rebaseline_20261116/official_full_uper512_fcn_20261120
/home/waas/mambavision_stage22/env/bin/python /home/featurize/work/mambavision_seg_rebaseline_20261116/four_dataset_matched_20261119/prepare_matched.py ISIC2018 PH2 Waterloo >matched_prepare.log 2>&1 &
pa=$!
/home/waas/mambavision_stage22/env/bin/python /home/featurize/work/mambavision_seg_rebaseline_20261116/ham10000_window16_20261119/prepare.py >ham_prepare.log 2>&1 &
pb=$!
(cd /home/featurize/work/mambavision_seg_rebaseline_20261116/uper512_ade_decoder_only_20261119 && /home/waas/mambavision_stage22/env/bin/python -c "from fetch_decoder import download; print(download(),flush=True)") >full_download.log 2>&1 &
pc=$!
curl -fsSL --retry 3 --max-time 80 'https://raw.githubusercontent.com/NVlabs/MambaVision/7860a506b2eb844eaaae676f08461ce8c3c26f43/semantic_segmentation/tools/mamba_vision.py' -o mamba_vision.py.fetch && mv mamba_vision.py.fetch mamba_vision.py
wait "$pa"
wait "$pb"
wait "$pc"
echo ready > prepared.ok
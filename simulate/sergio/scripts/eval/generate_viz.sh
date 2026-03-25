uv run python plot_eval_metrics.py \
        --models sergio_tgt spptv1_last_stgt nca_stgt \
        --state-runs /mnt/polished-lake/home/nkondapaneni/state_runs \
        --output ../results/metrics_vs_step_SERGIO_TGT_TST.png

uv run python plot_eval_metrics.py \
        --models rpnd_baseline spptv1_rpnd nca_rpnd \
        --state-runs /mnt/polished-lake/home/nkondapaneni/state_runs \
        --output ../results/metrics_vs_step_rpnd.png


uv run python plot_eval_metrics.py \
        --models rpnd_baseline spptv2_rpnd_fewshot reptile3k_rpnd_fewshot nca_rpnd_fewshot \
        --state-runs /mnt/polished-lake/home/nkondapaneni/state_runs \
        --output ../results/metrics_vs_step_rpnd_fewshot.png


uv run python plot_eval_metrics.py \
        --models sergio_tgtv2 spptv2_last_stgt nca_stgt_v2 \
        --state-runs /mnt/polished-lake/home/nkondapaneni/state_runs \
        --output ../results/metrics_vs_step_SERGIO_TGT_TST_v2.png
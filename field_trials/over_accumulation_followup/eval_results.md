# Eval Results

| pair_id | eval_dataset | merged_score | best_source | delta_vs_best | worst_source | delta_vs_worst | delta_vs_avg | source_score_method | historical_strategy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ca_family_sst2xIMDB_eval_imdb | imdb | 0.868 | 0.884 | -0.016 | 0.876 | -0.008 | -0.012 | native-task-proxy | audit_aware |
| ca_family_sst2xIMDB_eval_sst2 | sst2 | 0.880 | 0.884 | -0.004 | 0.876 | 0.004 | 0.000 | native-task-proxy | audit_aware |
| ca_family_sst2xIMDB_v2_eval_imdb | imdb | 0.870 | 0.902 | -0.032 | 0.856 | 0.014 | -0.009 | native-task-proxy | uniform_linear |
| ca_family_sst2xIMDB_v2_eval_sst2 | sst2 | 0.852 | 0.902 | -0.050 | 0.856 | -0.004 | -0.027 | native-task-proxy | uniform_linear |
| ca_retained_sst2_x_sst2 | sst2 | 0.876 | 0.902 | -0.026 | 0.884 | -0.008 | -0.017 | eval-matched | audit_aware |
| ca_same_imdb_x_imdb | imdb | 0.868 | 0.876 | -0.008 | 0.856 | 0.012 | 0.002 | eval-matched | uniform_linear |
| p2_control_agnews_x_mnli | ag_news | 0.938 | 0.938 | 0.000 | 0.842 | 0.096 | 0.048 | native-task-proxy | audit_aware |
| p2_control_hate_x_irony | tweet_eval/hate | 0.602 | 0.622 | -0.020 | 0.486 | 0.116 | 0.048 | native-task-proxy | uniform_linear |
| p2_retained_agnews | ag_news | 0.944 | 0.938 | 0.006 | 0.902 | 0.042 | 0.024 | eval-matched | norm_equalized |
| p3_control_agnews_x_emotion | ag_news | 0.870 | 0.912 | -0.042 | 0.772 | 0.098 | 0.028 | native-task-proxy | audit_aware |
| p3_control_sst2_x_agnews | sst2 | 0.838 | 0.912 | -0.074 | 0.886 | -0.048 | -0.061 | native-task-proxy | audit_aware |
| p3_nearmiss_hate | tweet_eval/hate | 0.598 | 0.520 | 0.078 | 0.426 | 0.172 | 0.125 | eval-matched | uniform_linear |
| p3_retained_agnews | ag_news | 0.894 | 0.912 | -0.018 | 0.884 | 0.010 | -0.004 | eval-matched | uniform_linear |
| p3_retained_sst2 | sst2 | 0.820 | 0.886 | -0.066 | 0.714 | 0.106 | 0.020 | eval-matched | norm_equalized |
| t01_cross_task_sst2_x_agnews | sst2 | 0.842 | 0.900 | -0.058 | 0.884 | -0.042 | -0.050 | native-task-proxy | audit_aware |
| t01_retained_sst2_x_sst2 | sst2 | 0.876 | 0.902 | -0.026 | 0.884 | -0.008 | -0.017 | eval-matched | audit_aware |
| t01_same_family_sst2_x_imdb | sst2 | 0.878 | 0.884 | -0.006 | 0.876 | 0.002 | -0.002 | native-task-proxy | audit_aware |
| t02_excluded_irony_x_hate | tweet_eval/irony | 0.604 | 0.614 | -0.010 | 0.444 | 0.160 | 0.075 | native-task-proxy | audit_aware |
| t02_near_miss_marginal_irony | tweet_eval/irony | 0.606 | 0.616 | -0.010 | 0.614 | -0.008 | -0.009 | eval-matched | audit_aware |
| t02_near_miss_substantial_hate | tweet_eval/hate | 0.606 | 0.540 | 0.066 | 0.444 | 0.162 | 0.114 | eval-matched | audit_aware |
| t02_retained_irony_x_irony | tweet_eval/irony | 0.606 | 0.614 | -0.008 | 0.600 | 0.006 | -0.001 | eval-matched | audit_aware |

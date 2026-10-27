from catboost import CatBoostRegressor

import gpu_explore_runner_new as runner


for config in runner.CONFIGS[-13:]:
    CatBoostRegressor(**runner.catboost_options(config, 42, 10))

print(f"validated {len(runner.CONFIGS[-13:])} follow-up configs")

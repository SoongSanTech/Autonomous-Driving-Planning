"""
GridSearchOrchestrator: 하이퍼파라미터 그리드 서치 자동화.

BC 하이퍼파라미터 및 RL Reward 가중치 조합을 자동으로 순회하며
학습 + 평가를 수행한다.
"""

import itertools
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GridSearchOrchestrator:
    """하이퍼파라미터 그리드 서치 자동화."""

    def __init__(self, experiment_logger, checkpoint_dir: str = "checkpoints",
                 device: str = "cuda"):
        self._logger = experiment_logger
        self.checkpoint_dir = checkpoint_dir
        self.device = device

    def _generate_combinations(self, param_grid: dict) -> list[dict]:
        """파라미터 그리드에서 모든 조합 생성.

        Args:
            param_grid: {"key": [val1, val2, ...], ...}

        Returns:
            모든 조합의 dict 리스트.

        Raises:
            ValueError: 빈 그리드.
        """
        if not param_grid:
            raise ValueError("파라미터 그리드가 비어있습니다")

        for key, values in param_grid.items():
            if not values:
                raise ValueError(f"파라미터 '{key}'의 값 리스트가 비어있습니다")

        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        combinations = []

        for combo in itertools.product(*value_lists):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def run_bc_grid_search(self, data_path: str, param_grid: dict) -> list[str]:
        """BC 하이퍼파라미터 그리드 서치 실행.

        Args:
            data_path: 학습 데이터 경로.
            param_grid: {"lr": [...], "batch_size": [...], ...}

        Returns:
            각 조합의 experiment_id 리스트.
        """
        combinations = self._generate_combinations(param_grid)
        experiment_ids = []
        failed_count = 0
        total = len(combinations)

        # 부모 실험 생성
        parent_id = self._logger.create_experiment(
            experiment_type="bc_grid_search",
            purpose=f"BC 그리드 서치 — {total}개 조합",
            config={"param_grid": str(param_grid), "data_path": data_path},
        )

        for i, combo in enumerate(combinations):
            # 50% 이상 실패 시 중단
            if failed_count > 0 and failed_count / (i) >= 0.5 and i >= 2:
                logger.warning(
                    "그리드 서치 중단: %d/%d 실패 (50%% 초과)", failed_count, i,
                )
                break

            eid = self._logger.create_experiment(
                experiment_type="bc_training",
                purpose=f"BC 그리드 서치 조합 {i + 1}/{total}",
                config=combo,
                parent_id=parent_id,
            )

            try:
                result = self._run_single_bc(data_path, combo)
                # 숫자 메트릭만 log_metrics에 전달 (문자열 제외)
                numeric_metrics = {k: v for k, v in result.items()
                                   if isinstance(v, (int, float))}
                self._logger.log_metrics(eid, numeric_metrics)
                self._logger.update_status(eid, "completed")
                experiment_ids.append(eid)

                # CLI 재현 명령어 기록
                args = " ".join(f"--{k} {v}" for k, v in combo.items())
                cmd = f"python -m model.train_bc --data_path {data_path} {args}"
                self._logger.log_cli_command(eid, cmd)

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    # GPU OOM: batch_size 절반 재시도
                    logger.warning("GPU OOM — batch_size 절반 재시도")
                    retry_combo = dict(combo)
                    if "batch_size" in retry_combo:
                        retry_combo["batch_size"] = max(1, retry_combo["batch_size"] // 2)
                    try:
                        result = self._run_single_bc(data_path, retry_combo)
                        numeric_metrics = {k: v for k, v in result.items()
                                           if isinstance(v, (int, float))}
                        self._logger.log_metrics(eid, numeric_metrics)
                        self._logger.update_status(eid, "completed")
                        experiment_ids.append(eid)
                    except Exception:
                        self._logger.update_status(eid, "failed")
                        failed_count += 1
                else:
                    self._logger.update_status(eid, "failed")
                    failed_count += 1
                    logger.error("조합 %d 실패: %s", i + 1, e)

            except Exception as e:
                self._logger.update_status(eid, "failed")
                failed_count += 1
                logger.error("조합 %d 실패: %s", i + 1, e)

        self._logger.update_status(parent_id, "completed")
        return experiment_ids

    def _run_single_bc(self, data_path: str, config: dict) -> dict:
        """단일 BC 학습 실행.

        실제 학습은 BCTrainer를 사용. 테스트에서는 모킹됨.
        """
        from model.bc_model import BehavioralCloningModel
        from model.bc_trainer import BCTrainer
        from model.dataset import DrivingDataset, default_transform
        from torch.utils.data import DataLoader, random_split

        lr = config.get("lr", 1e-4)
        batch_size = config.get("batch_size", 32)
        steering_weight = config.get("steering_weight", 2.0)
        epochs = config.get("epochs", 50)
        frozen_epochs = config.get("frozen_epochs", 10)

        dataset = DrivingDataset(data_path, transform=default_transform())
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = BehavioralCloningModel(pretrained=True)
        trainer = BCTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            steering_weight=steering_weight,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
        )

        result = trainer.train(epochs=epochs, frozen_epochs=frozen_epochs)
        return {
            "best_val_loss": result["best_val_loss"],
            "best_checkpoint": result.get("best_checkpoint", ""),
        }

    def run_rl_reward_grid_search(self, bc_checkpoint: str, reward_grid: dict,
                                   carla_host: str = "localhost") -> list[str]:
        """RL Reward 가중치 그리드 서치 실행.

        Args:
            bc_checkpoint: BC warm-start 체크포인트 경로.
            reward_grid: {"w_progress": [...], "w_collision": [...], ...}
            carla_host: CARLA 서버 호스트.

        Returns:
            각 조합의 experiment_id 리스트.
        """
        combinations = self._generate_combinations(reward_grid)
        experiment_ids = []
        failed_count = 0
        total = len(combinations)

        parent_id = self._logger.create_experiment(
            experiment_type="rl_reward_search",
            purpose=f"RL Reward 그리드 서치 — {total}개 조합",
            config={"reward_grid": str(reward_grid), "bc_checkpoint": bc_checkpoint},
        )

        for i, combo in enumerate(combinations):
            if failed_count > 0 and failed_count / (i) >= 0.5 and i >= 2:
                logger.warning(
                    "RL 그리드 서치 중단: %d/%d 실패 (50%% 초과)", failed_count, i,
                )
                break

            eid = self._logger.create_experiment(
                experiment_type="rl_training",
                purpose=f"RL Reward 그리드 서치 조합 {i + 1}/{total}",
                config=combo,
                parent_id=parent_id,
            )

            try:
                result = self._run_single_rl(bc_checkpoint, combo, carla_host)
                numeric_metrics = {k: v for k, v in result.items()
                                   if isinstance(v, (int, float))}
                self._logger.log_metrics(eid, numeric_metrics)
                self._logger.update_status(eid, "completed")
                experiment_ids.append(eid)

                args = " ".join(f"--{k} {v}" for k, v in combo.items())
                cmd = f"python -m model.train_rl --bc_checkpoint {bc_checkpoint} {args}"
                self._logger.log_cli_command(eid, cmd)

            except Exception as e:
                self._logger.update_status(eid, "failed")
                failed_count += 1
                logger.error("RL 조합 %d 실패: %s", i + 1, e)

        self._logger.update_status(parent_id, "completed")
        return experiment_ids

    def _run_single_rl(self, bc_checkpoint: str, reward_config: dict,
                        carla_host: str) -> dict:
        """단일 RL 학습 실행. 테스트에서는 모킹됨."""
        from model.carla_gym_env import CARLAGymEnv
        from model.checkpoint import CheckpointManager
        from model.reward import RewardFunction
        from model.rl_policy import RLPolicyNetwork
        from model.rl_trainer import RLTrainer

        policy = RLPolicyNetwork(pretrained=False)
        ckpt_mgr = CheckpointManager(self.checkpoint_dir)
        ckpt_mgr.load(bc_checkpoint, policy, device=self.device)

        env = CARLAGymEnv(host=carla_host)

        trainer = RLTrainer(
            policy=policy,
            env=env,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
        )

        result = trainer.train(num_episodes=1000)
        return {
            "best_avg_reward": result["best_avg_reward"],
            "best_checkpoint": result.get("best_checkpoint", ""),
        }

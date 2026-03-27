"""
실험 실행 CLI 엔트리포인트.

argparse 기반 서브커맨드: validate, train-bc, train-rl,
grid-search-bc, grid-search-rl, evaluate, report.
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment",
        description="CARLA 자율주행 실험 검증 CLI",
    )
    sub = parser.add_subparsers(dest="command", help="서브커맨드")

    # validate
    p_val = sub.add_parser("validate", help="데이터 품질 검증")
    p_val.add_argument("--session-dir", required=True, help="세션 디렉토리 경로")
    p_val.add_argument("--db-path", default="experiments/experiment_log.db")

    # train-bc
    p_bc = sub.add_parser("train-bc", help="BC 모델 학습")
    p_bc.add_argument("--data-path", required=True, help="학습 데이터 경로")
    p_bc.add_argument("--lr", type=float, default=1e-4)
    p_bc.add_argument("--batch-size", type=int, default=32)
    p_bc.add_argument("--epochs", type=int, default=50)
    p_bc.add_argument("--steering-weight", type=float, default=2.0)
    p_bc.add_argument("--frozen-epochs", type=int, default=10)
    p_bc.add_argument("--db-path", default="experiments/experiment_log.db")

    # train-rl
    p_rl = sub.add_parser("train-rl", help="RL 모델 학습")
    p_rl.add_argument("--bc-checkpoint", required=True, help="BC 체크포인트 경로")
    p_rl.add_argument("--num-episodes", type=int, default=5000)
    p_rl.add_argument("--carla-host", default="localhost")
    p_rl.add_argument("--db-path", default="experiments/experiment_log.db")

    # grid-search-bc
    p_gs_bc = sub.add_parser("grid-search-bc", help="BC 그리드 서치")
    p_gs_bc.add_argument("--data-path", required=True)
    p_gs_bc.add_argument("--db-path", default="experiments/experiment_log.db")

    # grid-search-rl
    p_gs_rl = sub.add_parser("grid-search-rl", help="RL Reward 그리드 서치")
    p_gs_rl.add_argument("--bc-checkpoint", required=True)
    p_gs_rl.add_argument("--carla-host", default="localhost")
    p_gs_rl.add_argument("--db-path", default="experiments/experiment_log.db")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="모델 평가")
    p_eval.add_argument("--checkpoint", required=True, help="모델 체크포인트 경로")
    p_eval.add_argument("--scenario", default=None, help="시나리오 ID (없으면 전체)")
    p_eval.add_argument("--num-runs", type=int, default=10)
    p_eval.add_argument("--carla-host", default="localhost")
    p_eval.add_argument("--db-path", default="experiments/experiment_log.db")

    # report
    p_report = sub.add_parser("report", help="실험 보고서 생성")
    p_report.add_argument("--db-path", default="experiments/experiment_log.db")
    p_report.add_argument("--output", default=None, help="보고서 출력 파일")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    from experiment.experiment_logger import ExperimentLogger

    logger = ExperimentLogger(db_path=args.db_path)

    if args.command == "validate":
        from experiment.data_validator import DataValidator
        validator = DataValidator(experiment_logger=logger)
        report = validator.validate_session(args.session_dir)
        print(f"검증 완료: {report.total_frames} 프레임, "
              f"{report.corrupted_frames} 손상, "
              f"재수집 필요: {report.needs_recollection}")

        eid = logger.create_experiment(
            "data_validation", "데이터 품질 검증",
            {"session_dir": args.session_dir},
        )
        logger.log_metrics(eid, {
            "total_frames": report.total_frames,
            "valid_frames": report.valid_frames,
            "corrupted_frames": report.corrupted_frames,
        })
        cmd = f"python -m experiment.cli validate --session-dir {args.session_dir}"
        logger.log_cli_command(eid, cmd)
        logger.update_status(eid, "completed")

    elif args.command == "train-bc":
        config = {
            "data_path": args.data_path,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "steering_weight": args.steering_weight,
            "frozen_epochs": args.frozen_epochs,
        }
        eid = logger.create_experiment("bc_training", "BC 학습", config)
        cmd_args = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in config.items())
        logger.log_cli_command(eid, f"python -m experiment.cli train-bc {cmd_args}")
        print(f"BC 학습 실험 생성: {eid}")
        # 실제 학습은 별도 실행

    elif args.command == "train-rl":
        config = {
            "bc_checkpoint": args.bc_checkpoint,
            "num_episodes": args.num_episodes,
            "carla_host": args.carla_host,
        }
        eid = logger.create_experiment("rl_training", "RL 학습", config)
        logger.log_cli_command(eid,
            f"python -m experiment.cli train-rl --bc-checkpoint {args.bc_checkpoint}")
        print(f"RL 학습 실험 생성: {eid}")

    elif args.command == "grid-search-bc":
        from experiment.grid_search import GridSearchOrchestrator
        orch = GridSearchOrchestrator(logger)
        param_grid = {
            "lr": [5e-5, 1e-4, 3e-4],
            "batch_size": [16, 32, 64],
            "steering_weight": [1.5, 2.0, 3.0],
        }
        ids = orch.run_bc_grid_search(args.data_path, param_grid)
        print(f"BC 그리드 서치 완료: {len(ids)}개 조합 성공")

    elif args.command == "grid-search-rl":
        from experiment.grid_search import GridSearchOrchestrator
        orch = GridSearchOrchestrator(logger)
        reward_grid = {
            "w_progress": [0.1, 0.3, 0.5],
            "w_collision": [0.5, 1.0, 2.0],
            "w_steering": [0.3, 0.5, 1.0],
        }
        ids = orch.run_rl_reward_grid_search(
            args.bc_checkpoint, reward_grid, carla_host=args.carla_host)
        print(f"RL 그리드 서치 완료: {len(ids)}개 조합 성공")

    elif args.command == "evaluate":
        print(f"평가 실행: checkpoint={args.checkpoint}, scenario={args.scenario}")

    elif args.command == "report":
        report_text = logger.generate_report()
        if args.output:
            with open(args.output, "w") as f:
                f.write(report_text)
            print(f"보고서 저장: {args.output}")
        else:
            print(report_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())

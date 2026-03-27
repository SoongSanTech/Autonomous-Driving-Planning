"""Unit tests for experiment CLI."""

import pytest

from experiment.cli import build_parser, main


class TestParser:
    def test_validate_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "--session-dir", "/tmp/session"])
        assert args.command == "validate"
        assert args.session_dir == "/tmp/session"

    def test_train_bc_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "train-bc", "--data-path", "/tmp/data",
            "--lr", "3e-4", "--batch-size", "64",
        ])
        assert args.command == "train-bc"
        assert args.data_path == "/tmp/data"
        assert args.lr == 3e-4
        assert args.batch_size == 64

    def test_train_bc_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["train-bc", "--data-path", "/tmp/data"])
        assert args.lr == 1e-4
        assert args.batch_size == 32
        assert args.epochs == 50
        assert args.steering_weight == 2.0
        assert args.frozen_epochs == 10

    def test_train_rl_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "train-rl", "--bc-checkpoint", "/tmp/bc.pth",
            "--num-episodes", "1000",
        ])
        assert args.command == "train-rl"
        assert args.bc_checkpoint == "/tmp/bc.pth"
        assert args.num_episodes == 1000

    def test_grid_search_bc_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["grid-search-bc", "--data-path", "/tmp/data"])
        assert args.command == "grid-search-bc"

    def test_grid_search_rl_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "grid-search-rl", "--bc-checkpoint", "/tmp/bc.pth",
        ])
        assert args.command == "grid-search-rl"

    def test_evaluate_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "evaluate", "--checkpoint", "/tmp/model.pth",
            "--scenario", "straight_clear_day", "--num-runs", "5",
        ])
        assert args.command == "evaluate"
        assert args.checkpoint == "/tmp/model.pth"
        assert args.scenario == "straight_clear_day"
        assert args.num_runs == 5

    def test_report_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["report", "--output", "/tmp/report.md"])
        assert args.command == "report"
        assert args.output == "/tmp/report.md"

    def test_no_command_returns_1(self):
        ret = main([])
        assert ret == 1

    def test_report_runs(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ret = main(["report", "--db-path", db_path])
        assert ret == 0

    def test_train_bc_creates_experiment(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ret = main([
            "train-bc", "--data-path", "/tmp/data",
            "--db-path", db_path,
        ])
        assert ret == 0

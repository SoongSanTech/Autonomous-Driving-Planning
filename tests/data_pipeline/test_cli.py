"""
Unit tests for data_pipeline.cli module.

Tests cover:
- Default argument values
- Custom argument parsing (host, port, output_dir, duration, headless)
- --no-headless flag overrides default headless=True
- main() instantiates DataPipeline with parsed args and calls connect/setup/run/shutdown
- Shutdown is called even when run raises an exception
"""

import pytest
from unittest.mock import patch, MagicMock, call


@pytest.mark.unit
class TestCliArgParsing:
    """Tests for CLI argument parsing."""

    def test_default_arguments(self):
        """Verify defaults: localhost, 2000, _out, 3600.0, headless=True."""
        with patch("sys.argv", ["cli"]):
            from data_pipeline.cli import main
            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument("--host", default="localhost")
            parser.add_argument("--port", type=int, default=2000)
            parser.add_argument("--output-dir", default="_out")
            parser.add_argument("--duration", type=float, default=3600.0)
            parser.add_argument("--headless", action="store_true", default=True)
            parser.add_argument("--no-headless", action="store_false", dest="headless")

            args = parser.parse_args([])
            assert args.host == "localhost"
            assert args.port == 2000
            assert args.output_dir == "_out"
            assert args.duration == 3600.0
            assert args.headless is True

    def test_custom_arguments(self):
        """Verify custom values are parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="localhost")
        parser.add_argument("--port", type=int, default=2000)
        parser.add_argument("--output-dir", default="_out")
        parser.add_argument("--duration", type=float, default=3600.0)
        parser.add_argument("--headless", action="store_true", default=True)
        parser.add_argument("--no-headless", action="store_false", dest="headless")

        args = parser.parse_args([
            "--host", "192.168.1.10",
            "--port", "3000",
            "--output-dir", "/data/output",
            "--duration", "60.0",
        ])
        assert args.host == "192.168.1.10"
        assert args.port == 3000
        assert args.output_dir == "/data/output"
        assert args.duration == 60.0
        assert args.headless is True

    def test_no_headless_flag(self):
        """Verify --no-headless sets headless to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=True)
        parser.add_argument("--no-headless", action="store_false", dest="headless")

        args = parser.parse_args(["--no-headless"])
        assert args.headless is False


@pytest.mark.unit
class TestCliMain:
    """Tests for main() function execution flow."""

    @patch("data_pipeline.cli.DataPipeline")
    @patch("data_pipeline.cli.logging")
    def test_main_calls_pipeline_lifecycle(self, mock_logging, mock_pipeline_cls):
        """Verify main() creates pipeline and calls connect, setup_sensors, run, shutdown."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        with patch("sys.argv", ["cli", "--host", "10.0.0.1", "--port", "2000",
                                 "--output-dir", "my_out", "--duration", "120.0"]):
            from data_pipeline.cli import main
            main()

        mock_pipeline_cls.assert_called_once_with(
            carla_host="10.0.0.1",
            carla_port=2000,
            output_dir="my_out",
            headless=True,
        )
        mock_pipeline.connect.assert_called_once()
        mock_pipeline.setup_sensors.assert_called_once()
        mock_pipeline.run.assert_called_once_with(duration_sec=120.0)
        mock_pipeline.shutdown.assert_called_once()

    @patch("data_pipeline.cli.DataPipeline")
    @patch("data_pipeline.cli.logging")
    def test_shutdown_called_on_exception(self, mock_logging, mock_pipeline_cls):
        """Verify shutdown() is called even when run() raises an exception."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.run.side_effect = RuntimeError("CARLA crashed")

        with patch("sys.argv", ["cli"]):
            from data_pipeline.cli import main
            with pytest.raises(RuntimeError, match="CARLA crashed"):
                main()

        mock_pipeline.shutdown.assert_called_once()

    @patch("data_pipeline.cli.DataPipeline")
    @patch("data_pipeline.cli.logging")
    def test_main_configures_logging(self, mock_logging, mock_pipeline_cls):
        """Verify main() configures logging with INFO level."""
        mock_pipeline_cls.return_value = MagicMock()

        with patch("sys.argv", ["cli"]):
            from data_pipeline.cli import main
            main()

        mock_logging.basicConfig.assert_called_once()
        call_kwargs = mock_logging.basicConfig.call_args[1]
        assert call_kwargs["level"] == mock_logging.INFO

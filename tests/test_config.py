import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from turbo_core.config import Settings, load_dotenv, turbo_home

# Python needs these to find the home folder, so tests that clear the
# environment keep them.
HOME_VARS = {k: os.environ[k] for k in ("USERPROFILE", "HOME", "HOMEDRIVE", "HOMEPATH") if k in os.environ}


class LoadDotenvTests(unittest.TestCase):
    def write_env(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return Path(handle.name)

    def test_parses_values_comments_and_quotes(self):
        path = self.write_env(
            "# comment\n\nTURBO_X_A=one\nexport TURBO_X_B=\"two words\"\nTURBO_X_C='3'\nnot a line\n"
        )
        with mock.patch.dict(os.environ, HOME_VARS, clear=True):
            load_dotenv(path)
            self.assertEqual(os.environ["TURBO_X_A"], "one")
            self.assertEqual(os.environ["TURBO_X_B"], "two words")
            self.assertEqual(os.environ["TURBO_X_C"], "3")

    def test_real_environment_wins(self):
        path = self.write_env("TURBO_X_A=from_file\n")
        with mock.patch.dict(os.environ, {"TURBO_X_A": "from_env"}, clear=True):
            load_dotenv(path)
            self.assertEqual(os.environ["TURBO_X_A"], "from_env")

    def test_missing_file_is_fine(self):
        load_dotenv(Path("does-not-exist.env"))


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.dict(os.environ, HOME_VARS, clear=True):
            settings = Settings.from_env(dotenv=False)
        self.assertEqual(settings.ollama_url, "http://localhost:11434")
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.token, "")
        self.assertTrue(settings.db_path.is_absolute())

    def test_overrides(self):
        env = {
            "TURBO_OLLAMA_URL": "http://box:1234/",
            "TURBO_CODE_MODEL": "m:1",
            "TURBO_PORT": "9000",
            "TURBO_TOKEN": "s3cret",
            "TURBO_DB_PATH": "C:/tmp/x.db",
        }
        with mock.patch.dict(os.environ, {**HOME_VARS, **env}, clear=True):
            settings = Settings.from_env(dotenv=False)
        self.assertEqual(settings.ollama_url, "http://box:1234")
        self.assertEqual(settings.code_model, "m:1")
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.token, "s3cret")
        self.assertEqual(settings.db_path, Path("C:/tmp/x.db"))

    def test_turbo_home_controls_db_and_dotenv_location(self):
        home = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(home, ignore_errors=True))
        (home / ".env").write_text("TURBO_CODE_MODEL=from-home-env\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {**HOME_VARS, "TURBO_HOME": str(home)}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.db_path, home / "turbo.db")
        self.assertEqual(settings.code_model, "from-home-env")

    def test_default_home_is_dot_turbo_in_user_folder(self):
        with mock.patch.dict(os.environ, HOME_VARS, clear=True):
            self.assertEqual(turbo_home(), Path.home() / ".turbo")

    def test_bad_number_gives_clear_error(self):
        with mock.patch.dict(os.environ, {**HOME_VARS, "TURBO_PORT": "abc"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TURBO_PORT"):
                Settings.from_env(dotenv=False)


if __name__ == "__main__":
    unittest.main()

"""数据集下载器单元测试（Part4K2 Task 8.5）。

覆盖用例：
1. DatasetDownloader 初始化
2. download_url：本地 HTTP 服务器下载（验证内容正确）
3. 断点续传：先写入部分文件，再 resume 完成下载
4. 禁用断点续传：已存在文件被覆盖
5. to_npz：JSON / JSONL / CSV / TXT 格式转换
6. download_and_cache：URL → .npz 一站式
7. HF datasets 不可用时优雅降级（mock datasets 不可用）
8. 多线程分块下载（>10MB 文件 + 4 workers）
9. 多线程幂等：完整文件再调用应直接返回
10. CLI verse-download --help / 无参数报错

注意：测试用本地 HTTP 服务器（支持 Range），不依赖真实网络。

运行方式：
    cd /workspace && python -m pytest tests/test_downloader.py -x -q
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

# PYTHONPATH 适配：让 verse_infra / verse_torch / verse_nex / data 可被 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_infra"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# 让 data.downloader 可被 import（namespace package，无需 __init__.py）
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)


# ---------------------------------------------------------------------------
# 本地 HTTP 服务器（支持 Range 请求，用于 download_url / 多线程测试）
# ---------------------------------------------------------------------------


class _RangeHTTPHandler(http.server.BaseHTTPRequestHandler):
    """支持 Range 请求的 HTTP handler，serve 指定目录。

    通过类属性 ``serve_directory`` 指定服务目录（由 _HTTPServerThread 注入）。
    """

    serve_directory: str = "."

    def log_message(self, *args, **kwargs):
        """静默日志（避免污染测试输出）。"""
        pass

    def _parse_range(self, range_header: str, file_size: int) -> Tuple[int, int]:
        """解析 ``Range: bytes=start-end`` 头。"""
        m = range_header.replace("bytes=", "").strip()
        if m.startswith("-"):
            # bytes=-N：最后 N 字节
            n = int(m[1:])
            return file_size - n, file_size - 1
        if m.endswith("-"):
            # bytes=start-：从 start 到末尾
            return int(m[:-1]), file_size - 1
        parts = m.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1
        return start, end

    def _send_file(self, head_only: bool = False):
        path = os.path.join(self.serve_directory, self.path.lstrip("/"))
        if not os.path.isfile(path):
            self.send_error(404, "Not Found")
            return
        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        with open(path, "rb") as f:
            if range_header:
                try:
                    start, end = self._parse_range(range_header, file_size)
                except (ValueError, IndexError):
                    self.send_error(400, "Bad Range")
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if not head_only:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(8192, remaining))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if not head_only:
                    shutil.copyfileobj(f, self.wfile, 8192)

    def do_GET(self):
        self._send_file(head_only=False)

    def do_HEAD(self):
        self._send_file(head_only=True)


class _HTTPServerThread(threading.Thread):
    """后台 HTTP 服务器线程。"""

    def __init__(self, directory: str):
        super().__init__(daemon=True)
        # 用 type() 动态创建子类，把 serve_directory 绑定为类属性
        handler = type(
            "BoundRangeHandler",
            (_RangeHTTPHandler,),
            {"serve_directory": directory},
        )
        # ThreadingTCPServer：支持并发请求（多线程下载需要）
        self.server = socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0), handler
        )
        self.server.daemon_threads = True
        self.port: int = self.server.server_address[1]
        self._stopped = False

    def run(self):
        try:
            self.server.serve_forever()
        except Exception:
            pass

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass


@pytest.fixture
def http_server(tmp_path):
    """启动本地 HTTP 服务器，serve tmp_path 目录。

    Yields:
        (port, serve_dir)
    """
    thread = _HTTPServerThread(str(tmp_path))
    thread.start()
    # 短暂等待服务器就绪
    import time
    time.sleep(0.05)
    try:
        yield thread.port, str(tmp_path)
    finally:
        thread.stop()


# ---------------------------------------------------------------------------
# 1. DatasetDownloader 初始化
# ---------------------------------------------------------------------------


def test_init(tmp_path):
    """DatasetDownloader 初始化：cache_dir 自动创建，参数正确保存。"""
    from data.downloader import DatasetDownloader
    cache = str(tmp_path / "cache")
    dl = DatasetDownloader(cache_dir=cache, num_workers=2, chunk_size=4096)
    assert dl.cache_dir == cache
    assert dl.num_workers == 2
    assert dl.chunk_size == 4096
    assert os.path.isdir(cache), "cache_dir 应被自动创建"


def test_init_default_params():
    """DatasetDownloader 默认参数。"""
    from data.downloader import DatasetDownloader
    dl = DatasetDownloader()
    assert dl.num_workers == 4
    assert dl.chunk_size == 8192
    assert os.path.isdir(dl.cache_dir)


# ---------------------------------------------------------------------------
# 2. download_url：本地 HTTP 下载
# ---------------------------------------------------------------------------


def test_download_url_basic(http_server, tmp_path):
    """download_url：从本地 HTTP 服务器下载小文件。"""
    port, serve_dir = http_server
    content = b"hello world\n" * 100
    src_file = Path(serve_dir) / "data.txt"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=1)
    out = dl.download_url(f"http://127.0.0.1:{port}/data.txt")
    assert os.path.isfile(out)
    with open(out, "rb") as f:
        assert f.read() == content


def test_download_url_explicit_output(http_server, tmp_path):
    """download_url：显式指定 output_path。"""
    port, serve_dir = http_server
    content = b"explicit output test\n"
    src_file = Path(serve_dir) / "data.bin"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=1)
    out_path = str(tmp_path / "custom" / "out.bin")
    result = dl.download_url(
        f"http://127.0.0.1:{port}/data.bin", output_path=out_path,
    )
    assert result == os.path.abspath(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == content


# ---------------------------------------------------------------------------
# 3. 断点续传
# ---------------------------------------------------------------------------


def test_download_url_resume(http_server, tmp_path):
    """断点续传：先写入部分内容，再 resume 完成。"""
    port, serve_dir = http_server
    # 16KB 文件（< 10MB 阈值，走单线程路径，便于断点续传测试）
    content = b"0123456789ABCDEF" * 1024  # 16KB
    src_file = Path(serve_dir) / "big.bin"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=1)
    out_path = str(tmp_path / "out.bin")

    # 先写入前 4KB（模拟上次中断）
    partial_size = 4096
    with open(out_path, "wb") as f:
        f.write(content[:partial_size])

    # 触发断点续传
    dl.download_url(
        f"http://127.0.0.1:{port}/big.bin",
        output_path=out_path, resume=True,
    )

    with open(out_path, "rb") as f:
        data = f.read()
    assert data == content, "断点续传后文件应与源文件完全一致"
    assert len(data) == len(content)


def test_download_url_no_resume_overwrites(http_server, tmp_path):
    """禁用断点续传：已存在文件会被覆盖（从头下载）。"""
    port, serve_dir = http_server
    content = b"hello\n"
    src_file = Path(serve_dir) / "data.txt"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=1)
    out_path = str(tmp_path / "out.txt")

    # 先写入垃圾数据
    with open(out_path, "wb") as f:
        f.write(b"garbage data that should be overwritten")

    dl.download_url(
        f"http://127.0.0.1:{port}/data.txt",
        output_path=out_path, resume=False,
    )

    with open(out_path, "rb") as f:
        assert f.read() == content


def test_download_url_resume_complete_noop(http_server, tmp_path):
    """resume=True 且文件已完整：直接返回，不重新下载。"""
    port, serve_dir = http_server
    content = b"complete file content\n"
    src_file = Path(serve_dir) / "data.txt"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=1)
    out_path = str(tmp_path / "out.txt")
    # 预先写入完整内容
    with open(out_path, "wb") as f:
        f.write(content)

    result = dl.download_url(
        f"http://127.0.0.1:{port}/data.txt",
        output_path=out_path, resume=True,
    )
    assert result == os.path.abspath(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == content


# ---------------------------------------------------------------------------
# 4. to_npz：格式转换
# ---------------------------------------------------------------------------


def test_to_npz_json(tmp_path):
    """to_npz：JSON 文件转换。"""
    items = [{"text": "hello"}, {"text": "world"}]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))
    npz = dl.to_npz(str(src))
    assert os.path.isfile(npz)

    data = np.load(npz, allow_pickle=False)
    assert "ids" in data.files
    assert "mask" in data.files
    assert "seq_len" in data.files
    assert "n_blocks" in data.files
    assert "source" in data.files
    assert int(data["seq_len"]) == 1
    assert int(data["n_blocks"]) == len(data["ids"])
    assert len(data["ids"]) > 0
    assert len(data["ids"]) == len(data["mask"])


def test_to_npz_jsonl(tmp_path):
    """to_npz：JSONL 文件转换。"""
    items = [{"text": "line1"}, {"text": "line2"}, {"text": "line3"}]
    src = tmp_path / "data.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))
    npz = dl.to_npz(str(src))
    data = np.load(npz, allow_pickle=False)
    assert len(data["ids"]) > 0


def test_to_npz_csv(tmp_path):
    """to_npz：CSV 文件转换。"""
    src = tmp_path / "data.csv"
    src.write_text("text,label\nhello,1\nworld,2\n", encoding="utf-8")

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))
    npz = dl.to_npz(str(src), text_key="text")
    data = np.load(npz, allow_pickle=False)
    assert len(data["ids"]) > 0


def test_to_npz_txt(tmp_path):
    """to_npz：TXT 文件转换（每行一条文本）。"""
    src = tmp_path / "data.txt"
    src.write_text("hello\nworld\nfoo bar\n", encoding="utf-8")

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))
    npz = dl.to_npz(str(src))
    data = np.load(npz, allow_pickle=False)
    assert len(data["ids"]) > 0


def test_to_npz_custom_text_key(tmp_path):
    """to_npz：自定义 text_key 字段名。"""
    items = [{"content": "abc"}, {"content": "def"}]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))
    npz = dl.to_npz(str(src), text_key="content")
    data = np.load(npz, allow_pickle=False)
    assert len(data["ids"]) > 0


def test_to_npz_explicit_output(tmp_path):
    """to_npz：显式指定 output_path。"""
    items = [{"text": "hello"}]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(items), encoding="utf-8")

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))
    out_npz = str(tmp_path / "sub" / "out.npz")
    result = dl.to_npz(str(src), output_path=out_npz)
    assert result == os.path.abspath(out_npz)
    assert os.path.isfile(out_npz)


# ---------------------------------------------------------------------------
# 5. download_and_cache：一站式
# ---------------------------------------------------------------------------


def test_download_and_cache_url(http_server, tmp_path):
    """download_and_cache：URL → .npz 一站式。"""
    port, serve_dir = http_server
    items = [{"text": "hello world"}, {"text": "foo bar"}]
    src = Path(serve_dir) / "data.json"
    src.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=1)
    npz_path = str(tmp_path / "out.npz")
    result = dl.download_and_cache(
        f"http://127.0.0.1:{port}/data.json",
        output_path=npz_path,
        text_key="text",
    )
    assert result == npz_path
    assert os.path.isfile(npz_path)
    data = np.load(npz_path, allow_pickle=False)
    assert len(data["ids"]) > 0


# ---------------------------------------------------------------------------
# 6. HF datasets 不可用时优雅降级
# ---------------------------------------------------------------------------


def test_download_hf_no_datasets(tmp_path, monkeypatch):
    """HF datasets 库不可用时：download_hf 抛 RuntimeError 提示安装。"""
    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))

    # 模拟 datasets 库不可用：sys.modules[name] = None 会让 import name 抛 ImportError
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(RuntimeError, match="datasets"):
        dl.download_hf("wikitext")


def test_download_hf_no_datasets_message(tmp_path, monkeypatch):
    """Runtime 错误信息应包含安装提示。"""
    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"))
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(RuntimeError) as exc_info:
        dl.download_hf("wikitext", split="train")
    assert "pip install datasets" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. 多线程分块下载
# ---------------------------------------------------------------------------


def test_download_multithread(http_server, tmp_path):
    """多线程分块下载：>10MB 文件 + 多线程，验证内容完整。"""
    port, serve_dir = http_server
    # 构造 12MB 文件（>10MB 阈值，触发多线程路径）
    content = b"ABCDEFGH" * (12 * 1024 * 1024 // 8)
    src_file = Path(serve_dir) / "big.bin"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=4)
    out = dl.download_url(f"http://127.0.0.1:{port}/big.bin")
    assert os.path.isfile(out)
    with open(out, "rb") as f:
        data = f.read()
    assert data == content
    assert len(data) == len(content)
    # 分片目录应已清理
    assert not os.path.isdir(out + ".parts")


def test_download_multithread_idempotent(http_server, tmp_path):
    """多线程下载完成后再次调用（resume=True）应幂等返回。"""
    port, serve_dir = http_server
    content = b"X" * (11 * 1024 * 1024)  # 11MB
    src_file = Path(serve_dir) / "big.bin"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=4)
    out = dl.download_url(f"http://127.0.0.1:{port}/big.bin")
    with open(out, "rb") as f:
        assert f.read() == content

    # 第二次调用：文件已完整，应直接返回
    out2 = dl.download_url(
        f"http://127.0.0.1:{port}/big.bin", resume=True,
    )
    assert out2 == out
    with open(out2, "rb") as f:
        assert f.read() == content


def test_download_multithread_two_workers(http_server, tmp_path):
    """多线程下载：2 workers 也能正确下载。"""
    port, serve_dir = http_server
    content = b"0123456789ABCDEF" * (11 * 1024 * 1024 // 16)  # 11MB
    src_file = Path(serve_dir) / "big.bin"
    src_file.write_bytes(content)

    from data.downloader import DatasetDownloader
    dl = DatasetDownloader(cache_dir=str(tmp_path / "cache"), num_workers=2)
    out = dl.download_url(f"http://127.0.0.1:{port}/big.bin")
    with open(out, "rb") as f:
        assert f.read() == content


# ---------------------------------------------------------------------------
# 8. CLI verse-download
# ---------------------------------------------------------------------------


def test_cli_download_help(capsys):
    """verse-download --help：返回 0 且打印用法。"""
    from verse_infra.verse_trainer.cli import main
    rc = main(["download", "--help"])
    out = capsys.readouterr()
    assert rc == 0
    # --help 由 argparse 输出到 stdout
    assert "verse-download" in out.out
    assert "--url" in out.out
    assert "--hf" in out.out


def test_cli_download_short_alias_help(capsys):
    """短别名 download --help 等价于 verse-download --help。"""
    from verse_infra.verse_trainer.cli import main
    rc = main(["download", "--help"])
    out = capsys.readouterr()
    assert rc == 0
    assert "verse-download" in out.out


def test_cli_download_no_args_errors(capsys):
    """verse-download 无 --url/--hf：报错退出（非 0）。"""
    from verse_infra.verse_trainer.cli import main
    rc = main(["download"])
    assert rc != 0
    err = capsys.readouterr().err
    # parser.error 会打印到 stderr
    assert "--url" in err or "--hf" in err


def test_cli_download_url_help(capsys):
    """verse-download 子命令的 --help 包含所有参数。"""
    from verse_infra.verse_trainer.cli import main
    rc = main(["verse-download", "--help"])
    out = capsys.readouterr().out
    assert rc == 0
    for flag in ("--url", "--hf", "--split", "--output", "-o", "--to-npz",
                 "--text-key", "--workers", "--no-resume"):
        assert flag in out, f"--help 输出应包含 {flag}"


# ---------------------------------------------------------------------------
# 9. verse_infra 顶层导出 DatasetDownloader
# ---------------------------------------------------------------------------


def test_verse_infra_export_dataset_downloader():
    """verse_infra 顶层应能导出 DatasetDownloader。"""
    # 这会触发 __getattr__，把 /workspace 加入 sys.path 并 import data.downloader
    from verse_infra import DatasetDownloader
    assert DatasetDownloader is not None
    assert hasattr(DatasetDownloader, "download_url")
    assert hasattr(DatasetDownloader, "download_hf")
    assert hasattr(DatasetDownloader, "to_npz")
    assert hasattr(DatasetDownloader, "download_and_cache")

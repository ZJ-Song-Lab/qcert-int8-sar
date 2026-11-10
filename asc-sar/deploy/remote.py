"""开发板远程工具: 走 SSH/SFTP 在 Atlas 200I DK 上执行命令与传文件。

用法:
    python deploy/remote.py run  --cmd "npu-smi info"
    python deploy/remote.py push --src deploy/acl_infer.py --dst /root/asc_sar/deploy/
    python deploy/remote.py pushdir --src datasets/ssdd/images/test --dst /root/asc_sar/data/images
    python deploy/remote.py pull --src /root/asc_sar/results/om_bench.json --dst results/

连接参数可用环境变量覆盖: ASC_HOST / ASC_PORT / ASC_USER / ASC_PASS
"""
import argparse
import os
import posixpath
from pathlib import Path

import paramiko

HOST = os.environ.get("ASC_HOST", "192.168.0.2")
PORT = int(os.environ.get("ASC_PORT", "22"))
USER = os.environ.get("ASC_USER", "root")
PASS = os.environ.get("ASC_PASS", "Mind@123")

# 板上已验证可用的 Python 依赖路径
BOARD_ENV = (
    "export PYTHONPATH=/root/atlas_exp/pylibs:"
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH; "
)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15, banner_timeout=120, auth_timeout=60)
    return c


def run(cmd, with_env=True, timeout=None):
    c = connect()
    try:
        full = (BOARD_ENV + cmd) if with_env else cmd
        _, out, err = c.exec_command(full, timeout=timeout, get_pty=False)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        code = out.channel.recv_exit_status()
        return code, o, e
    finally:
        c.close()


def _mkdir_p(sftp, remote_dir):
    parts, cur = remote_dir.strip("/").split("/"), ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def push(src, dst):
    src = Path(src)
    c = connect()
    try:
        sftp = c.open_sftp()
        if dst.endswith("/"):
            dst = posixpath.join(dst, src.name)
        _mkdir_p(sftp, posixpath.dirname(dst))
        sftp.put(str(src), dst)
        print(f"OK {src} -> {dst} ({src.stat().st_size} bytes)")
        sftp.close()
    finally:
        c.close()


def pushdir(src, dst):
    src = Path(src)
    files = [p for p in src.rglob("*") if p.is_file()]
    c = connect()
    try:
        sftp = c.open_sftp()
        _mkdir_p(sftp, dst)
        for i, p in enumerate(files, 1):
            rel = p.relative_to(src).as_posix()
            remote = posixpath.join(dst, rel)
            _mkdir_p(sftp, posixpath.dirname(remote))
            sftp.put(str(p), remote)
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")
        print(f"OK {len(files)} files -> {dst}")
        sftp.close()
    finally:
        c.close()


def pull(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    c = connect()
    try:
        sftp = c.open_sftp()
        sftp.get(src, str(dst))
        print(f"OK {src} -> {dst}")
        sftp.close()
    finally:
        c.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)

    r = sub.add_parser("run")
    r.add_argument("--cmd", required=True)
    r.add_argument("--no-env", action="store_true", help="不注入板上 PYTHONPATH")

    for name in ("push", "pushdir", "pull"):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True)
        p.add_argument("--dst", required=True)

    args = ap.parse_args()
    if args.action == "run":
        code, o, e = run(args.cmd, with_env=not args.no_env)
        print(o, end="")
        if e.strip():
            print("[stderr]", e, end="")
        print(f"[exit] {code}")
        raise SystemExit(code)
    elif args.action == "push":
        push(args.src, args.dst)
    elif args.action == "pushdir":
        pushdir(args.src, args.dst)
    else:
        pull(args.src, args.dst)


if __name__ == "__main__":
    main()

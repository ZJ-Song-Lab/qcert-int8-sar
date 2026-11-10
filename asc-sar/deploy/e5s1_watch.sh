#!/bin/bash
# E6 退出后自动启动 E5 设置1 板上流水线（避免空等；E5S1 与 E6 判定互不依赖）。
# 用法: setsid nohup bash e5s1_watch.sh <E6_PID> >/dev/null 2>&1 &
PID=$1
LOG=/root/asc_sar/code/results/e5s1_board.log
mkdir -p /root/asc_sar/code/results
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
sleep 15   # 等 E6 收尾写盘与释放 NPU/内存
echo "[watch] E6 (pid $PID) exited at $(date), starting e5s1_board.sh" >> "$LOG"
bash /root/asc_sar/deploy/e5s1_board.sh >> "$LOG" 2>&1
rc=$?
echo "[watch] e5s1_board.sh finished at $(date), exit=$rc" >> "$LOG"

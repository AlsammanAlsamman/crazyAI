@echo off
rem One-time resume of the hash gate+select batch (scheduled for 18:30 on 2026-10-01).
cd /d C:\Users\AlsammanA\Documents\mygithub\crazyAI
set "PATH=C:\Users\AlsammanA\tools\nodejs-portable;%PATH%"
set PYTHONPATH=.
set PYTHONIOENCODING=utf8
echo resume at %DATE% %TIME% >> archive\diagnose_return\provenance.txt
set SRC=invent_7201_hash invent_7202_hash invent_7203_hash invent_7204_hash invent_7221_hash invent_7222_hash invent_7223_hash invent_7224_hash invent_7225_hash
for %%k in (1 2 3) do start "crazyai gate k%%k" /min cmd /c "C:\Users\AlsammanA\AppData\Local\Programs\Python\Python312\python.exe -m crazyai.cli diagnose-return --sources %SRC% --variants c0_today__k%%k i3_gate__k%%k --provider claudecode --model claude-opus-5 --effort high --pin-model --timeout 2700 --max-calls 70 > archive\diagnose_return\run_gs4_k%%k.log 2>&1"

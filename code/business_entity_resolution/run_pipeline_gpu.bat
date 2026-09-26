@echo off
REM Full pipeline with a fine-tuned cross-encoder stacked into stage 2 (run on the CUDA machine after: pip install -r requirements.txt).
REM Expects the official files in  <challenge root>\dataset\train  and  <challenge root>\dataset\test
REM Resumable: finished cross-encoder fold groups are kept if this file is run again (earlier steps are recomputed).
REM Everything is logged to run_log_gpu.txt next to this file. Watch it live from another window with:
REM     powershell Get-Content run_log_gpu.txt -Wait -Tail 30
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set DATA=..\..\dataset
set LOG=run_log_gpu.txt
set PYTHONUNBUFFERED=1
type nul > "%LOG%"
echo Logging to %~dp0%LOG%
echo [%time%] 1/6 normalise the data
python -m src.data build --data-dir "%DATA%" --model-dir "models" >> "%LOG%" 2>&1 || goto :error
echo [%time%] 2/6 fine-tune the retrieval encoder (GPU)
python -m src.retrieval train --data-dir "%DATA%" --model-dir "models" --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 >> "%LOG%" 2>&1 || goto :error
echo [%time%] 3/6 retrieve candidates (GPU)
python -m src.retrieval index --data-dir "%DATA%" --model-dir "models" >> "%LOG%" 2>&1 || goto :error
echo [%time%] 4/6 features, stage 1, pruning
python -m src.main prepare --data-dir "%DATA%" --model-dir "models" >> "%LOG%" 2>&1 || goto :error
echo [%time%] 5/6 cross-encoder (GPU)
python -m src.cross_encoder --name mdeberta --backbone microsoft/mdeberta-v3-base --revision a0484667b22365f84929a935b5e50a51f71f159d --group-folds 2 --model-dir "models" >> "%LOG%" 2>&1 || goto :error
echo [%time%] 6/6 stage 2, decision, write output
python -m src.main stage2 --data-dir "%DATA%" --model-dir "models" --out-dir "..\..\output" --extra mdeberta --debug-dir "debug" >> "%LOG%" 2>&1 || goto :error
echo [%time%] official validator:
if exist "..\..\utils\validate_submission.py" (
  python "..\..\utils\validate_submission.py" --matching "..\..\output\matching_results.tsv" --candidate "..\..\output\candidate_pairs.tsv" --test-dir "%DATA%\test" > "%TEMP%\ber_validate.txt" 2>&1
  type "%TEMP%\ber_validate.txt"
  type "%TEMP%\ber_validate.txt" >> "%LOG%"
)
echo.
echo Done. Upload ..\..\output\matching_results.tsv to the leaderboard. The log is %~dp0%LOG%
pause
exit /b 0

:error
echo.
echo Pipeline stopped with an error (finished cross-encoder fold groups are kept). The last lines of %LOG%:
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
pause
exit /b 1

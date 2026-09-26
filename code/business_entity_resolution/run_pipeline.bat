@echo off
REM Full pipeline without the optional cross-encoder (run on the CUDA machine after: pip install -r requirements.txt).
REM normalise -> fine-tune bi-encoder -> retrieve candidates -> stage 1 + pruning -> stage 2 -> write + validate output\
REM Expects the official files in  <challenge root>\dataset\train  and  <challenge root>\dataset\test
REM Everything is logged to run_log.txt next to this file. Watch it live from another window with:
REM     powershell Get-Content run_log.txt -Wait -Tail 30
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set DATA=..\..\dataset
set LOG=run_log.txt
set PYTHONUNBUFFERED=1
type nul > "%LOG%"
echo Logging to %~dp0%LOG%
echo [%time%] 1/5 normalise the data
python -m src.data build --data-dir "%DATA%" --model-dir "models" >> "%LOG%" 2>&1 || goto :error
echo [%time%] 2/5 fine-tune the retrieval encoder (GPU)
python -m src.retrieval train --data-dir "%DATA%" --model-dir "models" --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 >> "%LOG%" 2>&1 || goto :error
echo [%time%] 3/5 retrieve candidates (GPU)
python -m src.retrieval index --data-dir "%DATA%" --model-dir "models" >> "%LOG%" 2>&1 || goto :error
echo [%time%] 4/5 features, stage 1, pruning
python -m src.main prepare --data-dir "%DATA%" --model-dir "models" >> "%LOG%" 2>&1 || goto :error
echo [%time%] 5/5 stage 2, decision, write output
python -m src.main stage2 --data-dir "%DATA%" --model-dir "models" --out-dir "..\..\output" >> "%LOG%" 2>&1 || goto :error
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
echo Pipeline stopped with an error. The last lines of %LOG%:
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
pause
exit /b 1

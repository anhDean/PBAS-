SETLOCAL ENABLEDELAYEDEXPANSION

set SCRIPTFOLDER=C:\Users\Dinh\Documents\GitHub\Master\Code\PBAS+\ConsoleApplication2\code\PBAS-\evaluation\scripts
set DATA_ROOT=E:\Datasets\datasets2012\dataset
set EVALFILE=%DATA_ROOT%\cm.txt
set MATLABFOLDER=%SCRIPTFOLDER%\Matlab
set PYTHONFOLDER=%SCRIPTFOLDER%\Python
set BATCHFOLDER=%SCRIPTFOLDER%\Batch
set OUTPUT_ROOT=E:\PBAS+2012
rem if output_root is changed, matlab scripts for evaluation must also be changed, dummyprocess function
set EVALCODEDIR=C:\Users\Dinh\Documents\GitHub\Master\Code\Matlab\EvaluationCode2012\
set EVAL_OUTPUT=E:\PBAS+2012_Eval\
set PROCESSOR=C:\Users\Dinh\Documents\GitHub\Master\Code\PBAS+\x64\Release\PBAS+.exe
set PARAMETERFILE=C:\Users\Dinh\Documents\GitHub\Master\Code\PBAS+\ConsoleApplication2\code\PBAS-\include\FrameProcessorParams.h


set PARAMTYPE=int
set PARAMNAME=N
set FIRSTRUN=1

set CSVFILE=%SCRIPTFOLDER%\..\data\%PARAMNAME%_eval_csv.dat

for /l %%x in (15, 5, 50) do (

set PARAMVAL=%%x

pushd %PYTHONFOLDER%
rem start /wait %BATCHFOLDER%\set_param.cmd "%PARAMETERFILE%" "(%PARAMTYPE%, %PARAMNAME%, !PARAMVAL!)"
popd

rem rebuild exe
rem start /wait rebuild_solution.cmd

rem process dataset
rem call %PROCESSOR% %DATA_ROOT% %OUTPUT_ROOT%

rem run evaluation
pushd %MATLABFOLDER%
rem start /wait %BATCHFOLDER%\evaluate_results.cmd %EVALCODEDIR% %DATA_ROOT% %EVAL_OUTPUT%
rem timeout 1800
popd

rem write eval results to csv file
pushd %PYTHONFOLDER%
start /wait %BATCHFOLDER%\write_csvfile.cmd "%EVALFILE%" "%PARAMNAME%" "!PARAMVAL!" "!FIRSTRUN!"
popd
set FIRSTRUN=0
)

rem generate plots
pushd %MATLABFOLDER%
matlab /r "filename='%CSVFILE%';generatePlotsFromCSV"
popd
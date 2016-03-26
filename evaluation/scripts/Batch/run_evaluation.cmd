@echo off
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

set PARAMTYPE=double
set PARAMNAME=defaultR

set PARAMTYPE2=double
set PARAMNAME2=RScale

set SOTA_RESULTS=%SCRIPTFOLDER%\..\init\state_of_the_art_csv.dat
set CSVFILE=%SCRIPTFOLDER%\..\data\%PARAMNAME%_eval_csv.dat

for /l %%k in (450, 50, 500) do (

set PARAMVAL2=%%k
rem use following code to pass floating point values
set /a DIV=!PARAMVAL2!/100
set /a MOD=!PARAMVAL2!%%100
rem echo !DIV!.!MOD!

set FIRSTRUN=1

pushd %PYTHONFOLDER%
start /wait %BATCHFOLDER%\set_param.cmd "%PARAMETERFILE%" "(%PARAMTYPE%, RScale, !DIV!.!MOD!)"
popd

for /l %%x in (16, 2, 20) do (

set PARAMVAL=%%x

pushd %PYTHONFOLDER%
start /wait %BATCHFOLDER%\set_param.cmd "%PARAMETERFILE%" "(%PARAMTYPE%, %PARAMNAME%, !PARAMVAL!)"
popd

rem rebuild exe
start /wait rebuild_solution.cmd

rem process dataset
call %PROCESSOR% %DATA_ROOT% %OUTPUT_ROOT%

rem run evaluation
pushd %MATLABFOLDER%
start /wait %BATCHFOLDER%\evaluate_results.cmd %EVALCODEDIR% %DATA_ROOT% %EVAL_OUTPUT%
timeout 1500
popd


rem write eval results to csv file
pushd %PYTHONFOLDER%
start /wait %BATCHFOLDER%\write_csvfile.cmd "%EVALFILE%" "%PARAMNAME%" "!PARAMVAL!" "!FIRSTRUN!"
popd

set FIRSTRUN=0
)

rem generate plots
pushd %MATLABFOLDER%
matlab /r "PARAM='%PARAMNAME%';filename='%CSVFILE%';sota_file='%SOTA_RESULTS%' ;generatePlotsFromCSV"
popd
timeout 100

rem move results in new eval folder
mkdir %SCRIPTFOLDER%\..\data\%PARAMNAME2%_%%k_%PARAMNAME%_!PARAMVAL!_eval_results_%DATE%
move  %SCRIPTFOLDER%\..\data\*.dat %SCRIPTFOLDER%\..\data\%PARAMNAME2%_%%k_%PARAMNAME%_!PARAMVAL!_eval_results_%DATE%
move  %SCRIPTFOLDER%\..\data\*.png %SCRIPTFOLDER%\..\data\%PARAMNAME2%_%%k_%PARAMNAME%_!PARAMVAL!_eval_results_%DATE%
copy  "%PARAMETERFILE%" "%SCRIPTFOLDER%\..\data\%PARAMNAME2%_%%k_%PARAMNAME%_!PARAMVAL!_eval_results_%DATE%\PBAS_parameter.txt"
copy "%EVALFILE%" "%SCRIPTFOLDER%\..\data\%PARAMNAME2%_%%k_%PARAMNAME%_!PARAMVAL!_eval_results_%DATE%\evaluation_results.txt"
)


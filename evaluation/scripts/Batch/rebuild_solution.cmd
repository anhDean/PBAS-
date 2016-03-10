set BUILD="C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe"
set SOLUTION="C:\Users\Dinh\Documents\GitHub\Master\Code\PBAS+\PBAS+.sln"
%BUILD% %SOLUTION% /Rebuild
timeout 3
exit
% Resume the reported runAblations failure from the completed CV checkpoint.
% Put this file and the patched main.m in your EXISTING project directory.
% Keep the original results directory. Do NOT run run_project again.
projectRoot=fileparts(mfilename('fullpath'));
addpath(projectRoot,'-begin');
clear main
resultFolder=fullfile(projectRoot,'results');
main(fullfile(projectRoot,'phoneme.arff'),resultFolder,true);
zip(fullfile(projectRoot,'MATLAB_rezultatai.zip'),{'results'},projectRoot);
fprintf('\nFinished. Upload MATLAB_rezultatai.zip for report review.\n');

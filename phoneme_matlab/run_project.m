% Open this script in MATLAB and press Run.
projectRoot=fileparts(mfilename('fullpath'));
addpath(projectRoot);
resultFolder=fullfile(projectRoot,'results');
main(fullfile(projectRoot,'phoneme.arff'),resultFolder);
zip(fullfile(projectRoot,'MATLAB_rezultatai.zip'),{'results'},projectRoot);
fprintf('\nCompleted. Upload MATLAB_rezultatai.zip for final report review.\n');

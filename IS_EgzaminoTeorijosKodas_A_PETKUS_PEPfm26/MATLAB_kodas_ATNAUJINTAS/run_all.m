%% VIENA KOMANDA VISAM EKSPERIMENTUI PAKARTOTI
% MATLAB Current Folder nustatykite i si aplanka ir paleiskite:
%   run_all
%
% Kiekvienas paleidimas gauna NAUJA aplanka runs/run_YYYYMMDD_HHMMSS.
% Ankstesni rezultatai neperrasomi, todel du paleidimus galima palyginti.

clearvars; clc; close all;
root=fileparts(mfilename('fullpath'));
runsRoot=fullfile(root,'runs');
if ~isfolder(runsRoot), mkdir(runsRoot); end

stamp=char(datetime('now','Format','yyyyMMdd_HHmmss'));
runName=['run_' stamp];
outDir=fullfile(runsRoot,runName);
idx=2;
while isfolder(outDir)
    runName=sprintf('run_%s_%02d',stamp,idx);
    outDir=fullfile(runsRoot,runName);
    idx=idx+1;
end

% main.m pats atsisius duomenis, jei phoneme.arff salia kodo nera.
main([],outDir,false);

% Issaugome tikslaus sio paleidimo kodo kopija.
snapshot=fullfile(outDir,'code_snapshot');
mkdir(snapshot);
copyfile(fullfile(root,'*.m'),snapshot);
for f={'README.md','AI_NAUDOJIMO_ZURNALAS.md'}
    src=fullfile(root,f{1});
    if isfile(src), copyfile(src,snapshot); end
end

% Pazymime naujausia paleidima, kad predictPhoneme veiktu be kelio argumento.
latestRun=outDir; %#ok<NASGU>
save(fullfile(root,'latest_run.mat'),'latestRun');
fid=fopen(fullfile(root,'LATEST_RUN.txt'),'w','n','UTF-8');
fprintf(fid,'%s\n',outDir); fclose(fid);

% Patogus archyvas pateikimui / pakartojamumo auditui.
zipFile=fullfile(runsRoot,[runName '.zip']);
zip(zipFile,{runName},runsRoot);

fprintf('\nBAIGTA. Rezultatai: %s\n',outDir);
fprintf('Archyvas: %s\n',zipFile);
fprintf('Pakartojimui dar karta paleiskite: run_all\n');

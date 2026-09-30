function main(dataFile,outDir,overwrite)
% MAIN Atkuriamas OpenML Phoneme (ID 1489) klasifikavimo eksperimentas.
%
% Rekomenduojamas pakartojimas: paleisti scenariju run_all.m.
% Jis kiekvienam bandymui sukuria nauja runs/run_... aplanka.
%
% Reikalavimai:
%   MATLAB R2023b (arba suderinama naujesne versija)
%   Statistics and Machine Learning Toolbox
%
% Pagrindiniai principai:
% 1) galutinis testas atskiriamas pries preprocessing ir modeliu derinima;
% 2) identiski pozymiu vektoriai pasalinami PRIES skaidyma, kad tas pats
%    vektorius negaletu patekti i mokymo ir testavimo imtis;
% 3) standartizavimo parametrai kiekvienoje kryzmines patikros dalyje
%    skaiciuojami tik is tos dalies mokymo poaibio;
% 4) hiperparametrai parenkami tik mokymo imties 5 daliu kryzmine patikra;
% 5) tikimybiu kalibravimas atliekamas tik atskiroje kalibravimo imtyje;
% 6) visu modeliu galutiniai palyginimo rodikliai skaiciuojami tame paciame
%    vienakart atvertame testavimo rinkinyje;
% 7) be klasikiniu modeliu igyvendinamas literaturos ikveptas
%    klaidų kainoms jautrus RBF-SVM (CS-RBF-SVM).

if nargin < 1 || isempty(dataFile)
    dataFile = fullfile(fileparts(mfilename('fullpath')),'phoneme.arff');
end
if nargin < 2 || isempty(outDir)
    outDir = fullfile(fileparts(mfilename('fullpath')),'results_revised');
end
if nargin < 3 || isempty(overwrite)
    overwrite = false;
end

assert(exist('fitcsvm','file')==2, ...
    'Reikalingas Statistics and Machine Learning Toolbox.');
assert(exist('fitcnet','file')==2, ...
    'Reikalinga MATLAB versija su fitcnet (R2021a ar naujesne).');

if ~isfile(dataFile)
    fprintf('Duomenu failas nerastas. Bandoma atsisiusti OpenML Phoneme...\n');
    downloadPhoneme(dataFile);
end

if isfolder(outDir)
    if overwrite
        rmdir(outDir,'s');
    else
        error(['Rezultatu aplankas jau egzistuoja: %s\n' ...
            'Jei norite pakartoti eksperimenta, treciajam argumentui perduokite true.'],outDir);
    end
end
mkdir(outDir);

diary(fullfile(outDir,'run_log.txt'));
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>
started = tic;

%% 1. Fiksuotas eksperimento protokolas
cfg = struct;
cfg.datasetID = 1489;
cfg.version = 'phoneme-revised-v2.1';
cfg.seed = 42;
cfg.folds = 5;
cfg.mlpSeeds = [11 22 33];
cfg.bootstrapSeed = 4242;
cfg.bootstrapN = 2000;
cfg.noiseLevels = [0 .10 .25 .50];
cfg.noiseSeeds = 101:110;
cfg.tieTolerance = .001;
cfg.iterationLimit = 1000; % atsarginė reikšmė; realiai MLP iteracijų limitas derinamas CV
cfg.fixedThreshold = .50;
cfg.testFraction = .20;
cfg.calFractionOfRemaining = .25; % 25 % is likusiu 80 % = apie 20 % visu duomenu
cfg.matlab = version;
cfg.release = version('-release');
cfg.toolboxes = ver;
cfg.primaryFamily = 'CS_SVM';
cfg.minDeltaBA = 0.02;
cfg.hypothesis = ['CS-RBF-SVM, parinktas tik train CV, testavimo imtyje turi ' ...
    'pasiekti bent 0.02 didesni subalansuota tiksluma uz stipresni atskaitos metoda, ' ...
    'porinio bootstrap 95% intervalo apatine riba turi buti > 0, o 1 klases ' ...
    'jautrumas (Recall1) neturi buti mazesnis uz atskaitos metodo.'];
cfg.literatureNote = ['2024 Zhu ir kt. Phoneme rinkiniui nagrinejo ' ...
    'klaidu kainoms jautru SVM ir MSHR-FCSSVM. Siame darbe igyvendinama ' ...
    'supaprastinta, aiskiai atskirta ideja: klaidingam mazosios klases ' ...
    'atmetimui priskiriama didesne klaidos kaina ir ji derinama tik train CV.'];
save(fullfile(outDir,'experiment_config.mat'),'cfg');

%% 2. Duomenu nuskaitymas ir semantikos patikra
[Xraw,yraw,manifest] = loadData(dataFile);
fprintf('Pradiniai duomenys: %d irasu, %d pozymiai. Klases 0/1: %d/%d.\n', ...
    size(Xraw,1),size(Xraw,2),sum(yraw==0),sum(yraw==1));

%% 3. Tiksliu dublikatu valdymas PRIES skaidyma
[X,y,dedup] = deduplicateFeatureVectors(Xraw,yraw);
manifest.rawN = size(Xraw,1);
manifest.n = size(X,1);
manifest.removedDuplicateFeatureVectors = dedup.removedRows;
manifest.conflictingDuplicateGroups = dedup.conflictingGroups;
manifest.classCountsAfterDedup = [sum(y==0),sum(y==1)];
save(fullfile(outDir,'data_manifest.mat'),'manifest','dedup');
writetable(dedup.table,fullfile(outDir,'deduplication_report.csv'));

%% 4. Stratifikuotas train / calibration / test skaidymas
rng(cfg.seed,'twister');
pTest = cvpartition(y,'HoldOut',cfg.testFraction);
splits.test = find(test(pTest));
remaining = find(training(pTest));
pCal = cvpartition(y(remaining),'HoldOut',cfg.calFractionOfRemaining);
splits.train = remaining(training(pCal));
splits.cal = remaining(test(pCal));

assert(isempty(intersect(splits.train,splits.cal)));
assert(isempty(intersect(splits.train,splits.test)));
assert(isempty(intersect(splits.cal,splits.test)));
assert(numel(unique([splits.train;splits.cal;splits.test]))==numel(y));

splitNames = ["train";"cal";"test"];
N = zeros(3,1); N0=N; N1=N;
for k=1:3
    ids=splits.(char(splitNames(k)));
    N(k)=numel(ids); N0(k)=sum(y(ids)==0); N1(k)=sum(y(ids)==1);
end
writetable(table(splitNames,N,N0,N1),fullfile(outDir,'split_counts.csv'));

%% 5. Duomenu nutekejimo auditas
leak = leakageAudit(X,splits);
assert(leak.TrainCalFeatureOverlap==0 && leak.TrainTestFeatureOverlap==0 && ...
    leak.CalTestFeatureOverlap==0, ...
    'Aptiktas identiskas pozymiu vektorius tarp skirtingu imciu. Eksperimentas stabdomas.');
writetable(struct2table(leak),fullfile(outDir,'leakage_audit.csv'));
save(fullfile(outDir,'splits.mat'),'splits');

%% 6. 5 daliu kryzmine patikra TIK mokymo imtyje
Xt = X(splits.train,:); yt = y(splits.train);
rng(cfg.seed,'twister');
cv = cvpartition(yt,'KFold',cfg.folds);
splits.fold=zeros(numel(yt),1);
for f=1:cfg.folds
    splits.fold(test(cv,f))=f;
end
save(fullfile(outDir,'splits.mat'),'splits','cv');

%% 7. Kandidatų tinklelis, įskaitant literatūros įkvėptą CS-RBF-SVM
[configs,labels,families] = makeGrid();
R = struct([]);
cvDetails = cell(numel(configs),1);
cvTimer=tic;
for q=1:numel(configs)
    c=configs{q};
    if strcmp(c.family,'MLP'), seeds=cfg.mlpSeeds; else, seeds=cfg.seed; end
    BA=zeros(cfg.folds,numel(seeds)); F1=BA; R1=BA;
    for f=1:cfg.folds
        idxTrain=training(cv,f); idxValid=test(cv,f);
        ppFold=fitPrep(Xt(idxTrain,:),true);
        Ztrain=applyPrep(Xt(idxTrain,:),ppFold);
        Zvalid=applyPrep(Xt(idxValid,:),ppFold);
        for s=1:numel(seeds)
            rng(seeds(s),'twister');
            model=trainOne(c,Ztrain,yt(idxTrain),cfg);
            pred=double(predict(model,Zvalid));
            met=classificationMetrics(yt(idxValid),pred);
            BA(f,s)=met.BA; F1(f,s)=met.F1; R1(f,s)=met.Recall1;
        end
    end
    R(q).Configuration=string(labels{q});
    R(q).Family=string(c.family);
    R(q).MeanBA=mean(BA(:));
    R(q).StdBA=std(BA(:));
    R(q).MeanF1=mean(F1(:));
    R(q).MeanRecall1=mean(R1(:));
    cvDetails{q}=struct('BA',BA,'F1',F1,'Recall1',R1);
    fprintf('CV %3d/%3d %-45s BA=%.5f F1=%.5f\n', ...
        q,numel(configs),labels{q},R(q).MeanBA,R(q).MeanF1);
end
cvSeconds=toc(cvTimer);
CV=struct2table(R);
writetable(CV,fullfile(outDir,'cv_results.csv'));
save(fullfile(outDir,'cv_details.mat'),'cvDetails','configs','cvSeconds');

%% 8. Hiperparametru pasirinkimas tik pagal train CV
familyOrder={'LR','kNN','SVM','CS_SVM','MLP'};
best=zeros(numel(familyOrder),1);
for k=1:numel(familyOrder)
    ii=find(strcmp(families,familyOrder{k}));
    bestBA=max(CV.MeanBA(ii));
    eligible=ii(CV.MeanBA(ii)>=bestBA-cfg.tieTolerance);
    % Jei BA praktiskai vienodas, renkamasi didesnis F1; jei ir jis vienodas,
    % pirmas tinklelio variantas (tinklelis sudetas nuo paprastesniu variantu).
    bestF1=max(CV.MeanF1(eligible));
    eligible=eligible(CV.MeanF1(eligible)>=bestF1-1e-12);
    best(k)=eligible(1);
end
selected=configs(best);
selectedCV=CV(best,:);
writetable(selectedCV,fullfile(outDir,'selected_parameters.csv'));

baselineFamilyIdx=1;
if CV.MeanBA(best(2))>CV.MeanBA(best(1)), baselineFamilyIdx=2; end
baselineConfigIdx=best(baselineFamilyIdx);

%% 9. Abliacija atliekama mokymo CV - testas vis dar neliestas
runAblations(Xt,yt,cv,selected{4},cfg,outDir);

%% 10. Galutinio preprocessing parametrai mokomi tik visoje train imtyje
pp=fitPrep(Xt,true);
Ztrain=applyPrep(Xt,pp);
Zcal=applyPrep(X(splits.cal,:),pp);
ycal=y(splits.cal);

%% 11. Galutiniai modeliai mokomi train imtyje
rng(cfg.seed,'twister'); modelLR=trainOne(selected{1},Ztrain,yt,cfg);
rng(cfg.seed,'twister'); modelKNN=trainOne(selected{2},Ztrain,yt,cfg);
rng(cfg.seed,'twister'); modelSVM=trainOne(selected{3},Ztrain,yt,cfg);
rng(cfg.seed,'twister'); modelCSSVM=trainOne(selected{4},Ztrain,yt,cfg);
modelMLP=cell(numel(cfg.mlpSeeds),1);
for s=1:numel(cfg.mlpSeeds)
    rng(cfg.mlpSeeds(s),'twister');
    modelMLP{s}=trainOne(selected{5},Ztrain,yt,cfg);
end

%% 12. Tikimybiu kalibravimas - tik cal imtyje
[calSVM,calInfoSVM]=fitPosterior(compact(modelSVM),Zcal,ycal);
[calCSSVM,calInfoCSSVM]=fitPosterior(compact(modelCSSVM),Zcal,ycal);
pCalSVM=positiveProbability(calSVM,Zcal);
pCalCSSVM=positiveProbability(calCSSVM,Zcal);
tSVM=chooseThresholdByBA(ycal,pCalSVM);
tCSSVM=chooseThresholdByBA(ycal,pCalCSSVM);

calibrationInfo=struct('SVMType',calInfoSVM.Type,'CSSVMType',calInfoCSSVM.Type, ...
    'SVMThreshold',tSVM,'CSSVMThreshold',tCSSVM);
save(fullfile(outDir,'calibration_info.mat'),'calibrationInfo');

%% 13. Protokolo uzrakinimas PRIES galutinio testo atverima
protocol=struct('selectedCV',selectedCV,'baselineConfigIdx',baselineConfigIdx, ...
    'primaryFamily',cfg.primaryFamily,'thresholdSVM',tSVM,'thresholdCSSVM',tCSSVM, ...
    'lockedAt',char(datetime('now')),'testMetricsComputed',false);
save(fullfile(outDir,'protocol_locked.mat'),'protocol','selected','pp','cfg');
writeProtocolLock(fullfile(outDir,'PROTOCOL_LOCKED.txt'),protocol,cfg);

%% 14. Vienkartinis visu modeliu palyginimas tame paciame testavimo rinkinyje
Ztest=applyPrep(X(splits.test,:),pp);
ytest=y(splits.test);

names=["LR";"kNN";"RBF_SVM";"CS_RBF_SVM";"MLP_ensemble3"];
H=zeros(numel(ytest),5); P=nan(numel(ytest),5);

H(:,1)=double(predict(modelLR,Ztest)); P(:,1)=positiveProbability(modelLR,Ztest);
H(:,2)=double(predict(modelKNN,Ztest)); P(:,2)=positiveProbability(modelKNN,Ztest);
H(:,3)=double(predict(modelSVM,Ztest)); P(:,3)=positiveProbability(calSVM,Ztest);
H(:,4)=double(predict(modelCSSVM,Ztest)); P(:,4)=positiveProbability(calCSSVM,Ztest);

pMlp=zeros(numel(ytest),numel(modelMLP));
for s=1:numel(modelMLP)
    pMlp(:,s)=positiveProbability(modelMLP{s},Ztest);
end
P(:,5)=mean(pMlp,2);
H(:,5)=double(P(:,5)>=cfg.fixedThreshold);

classRows=cell(5,1); probRows=cell(5,1);
for k=1:5
    classRows{k}=classificationMetrics(ytest,H(:,k));
    [~,~,~,auc]=perfcurve(ytest,P(:,k),1);
    probRows{k}=struct('Brier',mean((P(:,k)-ytest).^2),'AUC',auc);
end
TC=struct2table(vertcat(classRows{:}));
TC=addvars(TC,names,'Before',1,'NewVariableNames','Model');
writetable(TC,fullfile(outDir,'test_classification_metrics.csv'));
TP=struct2table(vertcat(probRows{:}));
TP=addvars(TP,names,'Before',1,'NewVariableNames','Model');
writetable(TP,fullfile(outDir,'test_probability_metrics.csv'));

%% 15. Kalibruoto 0.5 slenkscio ir cal dalyje parinkto slenkscio diagnostika
thresholdRows=cell(4,1);
thresholdRows{1}=thresholdMetricRow('RBF_SVM_calibrated_t0.5',ytest,P(:,3),.5);
thresholdRows{2}=thresholdMetricRow('RBF_SVM_calibrated_tCal',ytest,P(:,3),tSVM);
thresholdRows{3}=thresholdMetricRow('CS_RBF_SVM_calibrated_t0.5',ytest,P(:,4),.5);
thresholdRows{4}=thresholdMetricRow('CS_RBF_SVM_calibrated_tCal',ytest,P(:,4),tCSSVM);
Tthr=struct2table(vertcat(thresholdRows{:}));
writetable(Tthr,fullfile(outDir,'threshold_analysis_test.csv'));

%% 16. Hipotezes patikra: CS-RBF-SVM pries stipresni atskaitos metoda
baselineName=string(familyOrder{baselineFamilyIdx});
if baselineFamilyIdx==1, baselineCol=1; else, baselineCol=2; end
[deltaBA,ciBA,boot]=pairedBootstrapBA(ytest,H(:,4),H(:,baselineCol),cfg);
primary=classificationMetrics(ytest,H(:,4));
baseMet=classificationMetrics(ytest,H(:,baselineCol));
H1Confirmed=(deltaBA>=cfg.minDeltaBA) && (ciBA(1)>0) && (primary.Recall1>=baseMet.Recall1);
Hyp=table(baselineName,deltaBA,ciBA(1),ciBA(2),primary.Recall1-baseMet.Recall1,H1Confirmed, ...
    'VariableNames',{'Baseline','DeltaBA','CI95Low','CI95High','DeltaRecall1','H1Confirmed'});
writetable(Hyp,fullfile(outDir,'hypothesis_test.csv'));
save(fullfile(outDir,'bootstrap.mat'),'boot');

%% 17. Atsparumo triuksmui bandymas - modelis neperrenkamas
noiseStudy(modelCSSVM,calCSSVM,Ztest,ytest,cfg,outDir);

%% 18. Klaidų analize ir grafikai
errorAnalysis(X(splits.test,:),ytest,H(:,4),P(:,4),splits.test,manifest.featureNames,outDir);
plotModelComparison(TC,outDir);
plotConfusion(ytest,H(:,4),outDir);
calibrationPlot(ytest,P(:,4),outDir);
plotProbabilityVsTruth(ytest,P(:,4),outDir);

%% 19. Prognozavimo modelio paketas
bundle=struct;
bundle.rawModel=modelCSSVM;
bundle.calibratedModel=calCSSVM;
bundle.preprocess=pp;
bundle.featureNames=manifest.featureNames;
bundle.classNames=[0;1];
bundle.classSemantics={'0=nosinis','1=burninis'};
bundle.probabilityThreshold=tCSSVM;
bundle.labelRule='raw CS-RBF-SVM decision; probability from independent calibration split';
bundle.version=cfg.version;
bundle.configuration=selected{4};
save(fullfile(outDir,'model_primary.mat'),'bundle');

%% 20. Prognozavimo laikas ir pakartojamumo artefaktai
predictPhoneme(X(splits.test(1),:),bundle); % apšilimas
one=X(splits.test(1),:); batch=repmat(one,1000,1);
t=tic; for r=1:100, predictPhoneme(one,bundle); end; oneSeconds=toc(t)/100;
t=tic; for r=1:100, predictPhoneme(batch,bundle); end; batchSeconds=toc(t)/100;
modelFile=dir(fullfile(outDir,'model_primary.mat'));
perf=struct('TotalSeconds',toc(started),'CVSeconds',cvSeconds, ...
    'Predict1Seconds',oneSeconds,'Predict1000Seconds',batchSeconds, ...
    'ModelFileBytes',modelFile.bytes);
save(fullfile(outDir,'performance.mat'),'perf');

Pr=table(splits.test,ytest,'VariableNames',{'OriginalUniqueRow','TrueClass'});
for k=1:numel(names)
    Pr.(['Y_' char(names(k))])=H(:,k);
    Pr.(['P_' char(names(k))])=P(:,k);
end
writetable(Pr,fullfile(outDir,'test_predictions.csv'));

protocol.testMetricsComputed=true;
save(fullfile(outDir,'protocol_locked.mat'),'protocol','selected','pp','cfg');
writeRunSummary(outDir,manifest,dedup,leak,selectedCV,TC,TP,Hyp,Tthr,perf,cfg);
writeAutomaticReport(outDir,manifest,dedup,leak,selectedCV,TC,TP,Hyp,Tthr,perf,cfg);

fprintf('\nEksperimentas baigtas. Galutiniai testavimo rezultatai: %s\n', ...
    fullfile(outDir,'test_classification_metrics.csv'));
disp(TC);
disp(Hyp);
end

%% ========================================================================
function downloadPhoneme(path)
% Atsisiuntimo saltiniai pateikti prioriteto tvarka.
urls={ ...
    'https://openml.org/data/v1/download/1592281/phoneme.arff', ...
    'https://datahub.io/core/openml-datasets/_r/-/data/phoneme/phoneme.arff'};
lastErr='';
for k=1:numel(urls)
    try
        websave(path,urls{k});
        fprintf('Duomenys atsisiusti is %s\n',urls{k});
        return;
    catch e
        lastErr=e.message;
    end
end
error(['Nepavyko automatiskai atsisiusti phoneme.arff. Atsisiuskite OpenML 1489 ARFF ' ...
    'ir idekite salia main.m. Paskutine klaida: %s'],lastErr);
end

function [X,y,m]=loadData(path)
m=struct('datasetID',1489,'source','https://www.openml.org/d/1489', ...
    'sourceFile',char(path),'acquiredAt',char(datetime('now','TimeZone','UTC')));
assert(isfile(path),'Duomenu failas nerastas: %s',path);
[~,~,ext]=fileparts(path);
assert(strcmpi(ext,'.arff'),'Naudokite originalu OpenML 1489 ARFF faila.');

text=fileread(path); lines=splitlines(string(text)); names={}; dataStart=[];
for i=1:numel(lines)
    line=strtrim(char(lines(i)));
    if isempty(line)||startsWith(line,'%'), continue; end
    if startsWith(lower(line),'@attribute')
        tok=regexp(line,'(?i)^@attribute\s+("[^"]+"|''[^'']+''|\S+)\s+(.+)$','tokens','once');
        assert(~isempty(tok),'Neatpazinta ARFF atributo eilute.');
        names{end+1}=regexprep(tok{1},'^[''"]|[''"]$',''); %#ok<AGROW>
    elseif strcmpi(line,'@data')
        dataStart=i+1; break;
    end
end
assert(~isempty(dataStart)&&numel(names)==6,'Tikimasi 6 ARFF stulpeliu.');
target=find(strcmpi(names,'Class'));
assert(numel(target)==1,'Nepavyko vienareiksmiskai rasti Class stulpelio.');

rows=nan(numel(lines)-dataStart+1,6); n=0;
for i=dataStart:numel(lines)
    line=strtrim(char(lines(i)));
    if isempty(line)||startsWith(line,'%'), continue; end
    assert(~startsWith(line,'{'),'Sparse ARFF nepalaikomas.');
    fields=strsplit(line,',','CollapseDelimiters',false);
    assert(numel(fields)==6,'Neteisingas lauku skaicius ARFF eiluteje %d.',i);
    n=n+1;
    for j=1:6
        token=regexprep(strtrim(fields{j}),'^[''"]|[''"]$','');
        if strcmp(token,'?')
            rows(n,j)=NaN;
        else
            rows(n,j)=str2double(token);
            assert(isfinite(rows(n,j)),'Neskaitine reiksme ARFF eiluteje %d.',i);
        end
    end
end
rows=rows(1:n,:);
features=setdiff(1:6,target,'stable');
X=rows(:,features); y=rows(:,target);
m.originalLabels=unique(y)';
if isequal(unique(y),[1;2])
    m.labelMapping=[1 0;2 1]; y=y-1;
elseif isequal(unique(y),[0;1])
    m.labelMapping=[0 0;1 1];
else
    error('Nezinomos klasiu zymos; automatinis spejimas draudziamas.');
end
assert(all(ismember(y,[0 1]))&&numel(unique(y))==2);
assert(size(X,1)==5404,'Netiketas irasu skaicius. Patikrinkite duomenu versija.');
assert(sum(y==0)==3818 && sum(y==1)==1586,'Netiketas klasiu pasiskirstymas.');
m.featureNames=names(features);
m.classSemantics={'0=nasal (nosinis)','1=oral (burninis)'};
m.nRaw=n;
m.classCountsRaw=[sum(y==0),sum(y==1)];
m.missingPerFeature=sum(isnan(X),1);
m.duplicateExactRows=n-size(unique([X y],'rows'),1);
m.duplicateFeatureRows=n-size(unique(X,'rows'),1);
end

function [X2,y2,d]=deduplicateFeatureVectors(X,y)
% Vienodi pozymiu vektoriai negali likti skirtingose imtyse.
[X2,ia,grp]=unique(X,'rows','stable');
y2=zeros(size(X2,1),1);
conflict=false(size(X2,1),1);
count=zeros(size(X2,1),1);
for g=1:size(X2,1)
    ids=find(grp==g); labels=unique(y(ids)); count(g)=numel(ids);
    if numel(labels)~=1
        conflict(g)=true;
    else
        y2(g)=labels(1);
    end
end
assert(~any(conflict), ...
    'Rasti identiski pozymiu vektoriai su skirtingomis klasemis. Reikalinga atskira duomenu analize.');
removed=size(X,1)-size(X2,1);
d=struct;
d.removedRows=removed;
d.conflictingGroups=sum(conflict);
d.representativeOriginalRows=ia;
d.table=table((1:size(X2,1))',ia,count,y2, ...
    'VariableNames',{'UniqueRow','RepresentativeOriginalRow','OriginalCount','Class'});
end

function a=leakageAudit(X,s)
a=struct;
a.TrainCalIndexOverlap=numel(intersect(s.train,s.cal));
a.TrainTestIndexOverlap=numel(intersect(s.train,s.test));
a.CalTestIndexOverlap=numel(intersect(s.cal,s.test));
a.TrainCalFeatureOverlap=countRowOverlap(X(s.train,:),X(s.cal,:));
a.TrainTestFeatureOverlap=countRowOverlap(X(s.train,:),X(s.test,:));
a.CalTestFeatureOverlap=countRowOverlap(X(s.cal,:),X(s.test,:));
a.PreprocessingLearnedFrom='train only; fold-local inside CV';
a.HyperparametersLearnedFrom='train 5-fold CV only';
a.CalibrationLearnedFrom='calibration split only';
a.FinalMetricsFrom='held-out test only';
end

function n=countRowOverlap(A,B)
if isempty(A)||isempty(B), n=0; return; end
n=size(intersect(A,B,'rows'),1);
end

function pp=fitPrep(X,standardize)
pp.median=median(X,1,'omitnan');
assert(all(isfinite(pp.median)),'Visas mokymo pozymio stulpelis yra NaN.');
for j=1:size(X,2), X(isnan(X(:,j)),j)=pp.median(j); end
pp.mu=mean(X,1); pp.sigma=std(X,0,1); pp.sigma(pp.sigma==0)=1;
pp.standardize=standardize;
end

function Z=applyPrep(X,pp)
assert(~any(isinf(X(:))),'Ivestyje yra Inf.');
for j=1:size(X,2), X(isnan(X(:,j)),j)=pp.median(j); end
Z=X;
if pp.standardize, Z=(X-pp.mu)./pp.sigma; end
assert(all(isfinite(Z(:))),'Po paruosimo liko nebaigtiniu reiksmiu.');
end

function [C,L,F]=makeGrid()
C={}; L={}; F={};
% Paprastesni variantai dedami pirmiau, kad vienodo rezultato atveju butu
% pasirenkamas paprastesnis modelis.
for lambda=[1 .01 .0001]
    add(struct('family','LR','lambda',lambda),sprintf('LR lambda=%g',lambda));
end
for k=[21 11 5 3]
    for w={'equal','inverse'}
        add(struct('family','kNN','k',k,'weight',w{1}),sprintf('kNN k=%d weight=%s',k,w{1}));
    end
end
for box=[.1 1 10]
    for gamma=[.01 .1 1 10]
        for prior={'empirical','uniform'}
            add(struct('family','SVM','C',box,'gamma',gamma,'prior',prior{1}), ...
                sprintf('SVM C=%g gamma=%g prior=%s',box,gamma,prior{1}));
        end
    end
end
% Literaturos ikveptas klaidų kainoms jautrus RBF-SVM. Didesnis Cost(2,1)
% labiau baudzia burninio (1) garso priskyrima nosiniam (0).
for box=[1 10]
    for gamma=[.1 1 10]
        for fnCost=[1.25 1.5 2 3]
            add(struct('family','CS_SVM','C',box,'gamma',gamma,'fnCost',fnCost), ...
                sprintf('CS-SVM C=%g gamma=%g FNcost=%g',box,gamma,fnCost));
        end
    end
end
for layers={16,32,[32 16]}
    for lambda=[.01 .0001]
        for iterLimit=[300 1000]
            add(struct('family','MLP','layers',layers{1},'lambda',lambda, ...
                'iterLimit',iterLimit), ...
                sprintf('MLP layers=%s lambda=%g iter=%d', ...
                mat2str(layers{1}),lambda,iterLimit));
        end
    end
end
    function add(c,label)
        C{end+1}=c; L{end+1}=label; F{end+1}=c.family;
    end
end

function m=trainOne(c,X,y,cfg)
switch c.family
    case 'LR'
        m=fitclinear(X,y,'Learner','logistic','Regularization','ridge', ...
            'Lambda',c.lambda,'Solver','lbfgs','ClassNames',[0;1]);
    case 'kNN'
        m=fitcknn(X,y,'NumNeighbors',c.k,'Distance','euclidean', ...
            'DistanceWeight',c.weight,'Standardize',false,'ClassNames',[0;1]);
    case 'SVM'
        m=fitcsvm(X,y,'KernelFunction','rbf','BoxConstraint',c.C, ...
            'KernelScale',1/sqrt(c.gamma),'Prior',c.prior, ...
            'Standardize',false,'ClassNames',[0;1]);
    case 'CS_SVM'
        cost=[0 1; c.fnCost 0];
        m=fitcsvm(X,y,'KernelFunction','rbf','BoxConstraint',c.C, ...
            'KernelScale',1/sqrt(c.gamma),'Cost',cost, ...
            'Standardize',false,'ClassNames',[0;1]);
    case 'MLP'
        m=fitcnet(X,y,'LayerSizes',c.layers,'Activations','relu', ...
            'Lambda',c.lambda,'Standardize',false,'ClassNames',[0;1], ...
            'IterationLimit',c.iterLimit);
    otherwise
        error('Nezinoma modelio seima: %s',c.family);
end
end

function p=positiveProbability(m,X)
[~,score]=predict(m,X);
idx=find(m.ClassNames==1);
assert(numel(idx)==1,'Modelyje nerasta teigiama 1 klase.');
p=score(:,idx);
assert(all(isfinite(p)),'Tikimybiu masyve yra NaN/Inf.');
% Tik kalibruoti SVM ir probabilistiniai klasifikatoriai turi buti [0,1].
if any(p < -1e-8 | p > 1+1e-8)
    error('Gauti score nera tikimybes. Patikrinkite, ar SVM buvo kalibruotas.');
end
p=min(1,max(0,p));
end

function m=classificationMetrics(y,h)
assert(numel(y)==numel(h)&&all(ismember(h,[0 1])));
tp=sum(y==1 & h==1); tn=sum(y==0 & h==0);
fp=sum(y==0 & h==1); fn=sum(y==1 & h==0);
rec1=safeDivide(tp,tp+fn); rec0=safeDivide(tn,tn+fp);
prec1=safeDivide(tp,tp+fp);
m=struct('TN',tn,'FP',fp,'FN',fn,'TP',tp, ...
    'Accuracy',(tp+tn)/numel(y),'Recall0',rec0,'Recall1',rec1, ...
    'Precision1',prec1,'BA',(rec0+rec1)/2, ...
    'F1',safeDivide(2*tp,2*tp+fp+fn));
end

function z=safeDivide(a,b)
if b==0, z=0; else, z=a/b; end
end

function t=chooseThresholdByBA(y,p)
grid=0.05:0.01:0.95;
ba=zeros(size(grid));
for i=1:numel(grid)
    ba(i)=classificationMetrics(y,double(p>=grid(i))).BA;
end
best=max(ba);
ids=find(ba>=best-1e-12);
[~,j]=min(abs(grid(ids)-.5));
t=grid(ids(j));
end

function r=thresholdMetricRow(name,y,p,t)
m=classificationMetrics(y,double(p>=t));
r=struct('Model',string(name),'Threshold',t,'BA',m.BA,'F1',m.F1, ...
    'Recall1',m.Recall1,'Accuracy',m.Accuracy);
end

function [delta,ci,boot]=pairedBootstrapBA(y,a,b,cfg)
ma=classificationMetrics(y,a); mb=classificationMetrics(y,b); delta=ma.BA-mb.BA;
i0=find(y==0); i1=find(y==1);
rng(cfg.bootstrapSeed,'twister'); boot=zeros(cfg.bootstrapN,1);
for r=1:cfg.bootstrapN
    ids=[i0(randi(numel(i0),numel(i0),1)); i1(randi(numel(i1),numel(i1),1))];
    ma=classificationMetrics(y(ids),a(ids)); mb=classificationMetrics(y(ids),b(ids));
    boot(r)=ma.BA-mb.BA;
end
ci=prctile(boot,[2.5 97.5]);
end

function runAblations(X,y,cv,c,cfg,outDir)
labels=["control";"A1_no_standardization";"A2_drop_1";"A2_drop_2"; ...
    "A2_drop_3";"A2_drop_4";"A2_drop_5";"A3_cost_equal_1"];
values=zeros(numel(labels),cfg.folds);
for q=1:numel(labels)
    keep=1:5; standardize=true; cq=c;
    if q==2, standardize=false; end
    if q>=3 && q<=7, keep(q-2)=[]; end
    if q==8, cq.fnCost=1; end
    for f=1:cfg.folds
        a=training(cv,f); b=test(cv,f);
        pp=fitPrep(X(a,keep),standardize);
        Za=applyPrep(X(a,keep),pp); Zb=applyPrep(X(b,keep),pp);
        rng(cfg.seed,'twister'); model=trainOne(cq,Za,y(a),cfg);
        met=classificationMetrics(y(b),double(predict(model,Zb)));
        values(q,f)=met.BA;
    end
end
T=table(labels,mean(values,2),std(values,0,2),mean(values(1,:))-mean(values,2), ...
    'VariableNames',{'Variant','MeanBA','StdBA','DropFromControl'});
writetable(T,fullfile(outDir,'ablation_train_cv.csv'));
end

function noiseStudy(rawModel,calModel,Z,y,cfg,outDir)
rows=cell(numel(cfg.noiseLevels)*numel(cfg.noiseSeeds),1); n=0;
summary=cell(numel(cfg.noiseLevels),1);
clean=classificationMetrics(y,double(predict(rawModel,Z)));
for li=1:numel(cfg.noiseLevels)
    level=cfg.noiseLevels(li); ba=zeros(numel(cfg.noiseSeeds),1); f1=ba;
    for r=1:numel(cfg.noiseSeeds)
        rng(cfg.noiseSeeds(r),'twister'); Zn=Z+level*randn(size(Z));
        h=double(predict(rawModel,Zn)); met=classificationMetrics(y,h);
        ba(r)=met.BA; f1(r)=met.F1; n=n+1;
        rows{n}=struct('Sigma',level,'Seed',cfg.noiseSeeds(r),'BA',met.BA,'F1',met.F1);
    end
    summary{li}=struct('Sigma',level,'MeanBA',mean(ba),'StdBA',std(ba), ...
        'MeanF1',mean(f1),'StdF1',std(f1),'MeanBADrop',clean.BA-mean(ba));
end
writetable(struct2table(vertcat(rows{:})),fullfile(outDir,'noise_repetitions.csv'));
T=struct2table(vertcat(summary{:}));
writetable(T,fullfile(outDir,'noise_summary.csv'));
f=figure('Visible','off'); errorbar(T.Sigma,T.MeanBA,T.StdBA,'o-','LineWidth',1.4);
xlabel('Triukšmo standartinis nuokrypis'); ylabel('Subalansuotas tikslumas');
grid on; title('CS-RBF-SVM atsparumas požymių triukšmui');
saveas(f,fullfile(outDir,'noise_primary.png')); close(f);
% calModel argumentas paliktas tycia: triuksmo klasifikavimo metrika skaiciuojama
% pagal uzrakinta neapdoroto SVM sprendimo taisykle, o ne pagal naujai derinama slenksti.
if isempty(calModel), error('Kalibruotas modelis netiketai tuscias.'); end
end

function errorAnalysis(X,y,h,p,ids,featureNames,outDir)
kind=repmat("TN",numel(y),1); kind(y==1 & h==1)="TP";
kind(y==0 & h==1)="FP"; kind(y==1 & h==0)="FN";
wrong=h~=y; near=abs(p-.5)<=.05;
high=wrong & ((h==1 & p>.9)|(h==0 & p<.1));
T=table(ids,y,h,p,kind,near,high,'VariableNames', ...
    {'UniqueRow','TrueClass','PredictedClass','Probability1','Outcome','NearHalf','HighConfidenceError'});
for j=1:size(X,2), T.(sprintf('Feature%d',j))=X(:,j); end
writetable(T,fullfile(outDir,'error_analysis_all.csv'));
writetable(T(wrong|near|high,:),fullfile(outDir,'error_examples.csv'));
rows=cell(20,1); r=0;
for g=["TN","FP","FN","TP"]
    use=kind==g;
    for j=1:5
        r=r+1; v=X(use,j);
        rows{r}=struct('Outcome',g,'Feature',string(featureNames{j}), ...
            'Count',numel(v),'Mean',mean(v,'omitnan'),'Median',median(v,'omitnan'), ...
            'Std',std(v,'omitnan'));
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(outDir,'error_feature_statistics.csv'));
end

function calibrationPlot(y,p,outDir)
bin=min(floor(p*10)+1,10); rows=cell(10,1);
for b=1:10
    use=bin==b;
    if any(use)
        mp=mean(p(use)); of=mean(y(use));
    else
        mp=NaN; of=NaN;
    end
    rows{b}=struct('Bin',b,'Lower',(b-1)/10,'Upper',b/10,'Count',sum(use), ...
        'MeanProbability',mp,'ObservedFraction',of);
end
T=struct2table(vertcat(rows{:}));
writetable(T,fullfile(outDir,'calibration_bins_primary.csv'));
f=figure('Visible','off'); plot([0 1],[0 1],'k--'); hold on;
ok=T.Count>0; plot(T.MeanProbability(ok),T.ObservedFraction(ok),'o-','LineWidth',1.5);
xlabel('Vidutinė prognozuota P(1)'); ylabel('Faktinė 1 klasės dalis');
axis([0 1 0 1]); grid on; title('CS-RBF-SVM kalibravimo kreivė');
saveas(f,fullfile(outDir,'calibration_primary.png')); close(f);
end

function plotModelComparison(T,outDir)
f=figure('Visible','off','Position',[100 100 900 500]);
bar(categorical(T.Model),[T.BA T.F1]); ylim([0 1]); grid on;
ylabel('Rodiklio reikšmė'); legend({'BA','F1'},'Location','southoutside','Orientation','horizontal');
title('Modelių rezultatai atskiroje testavimo imtyje');
saveas(f,fullfile(outDir,'test_model_comparison.png')); close(f);
end

function plotConfusion(y,h,outDir)
C=confusionmat(y,h,'Order',[0 1]);
f=figure('Visible','off'); confusionchart(C,{'0 - nosinis','1 - burninis'});
title('CS-RBF-SVM sumaišties matrica: testavimo imtis');
saveas(f,fullfile(outDir,'confusion_primary.png')); close(f);
end

function plotProbabilityVsTruth(y,p,outDir)
[ps,ord]=sort(p);
ys=y(ord);
f=figure('Visible','off','Position',[100 100 950 500]);
plot(ps,'LineWidth',1.2); hold on;
scatter(1:numel(ys),ys,9,'filled');
yline(.5,'--'); grid on; ylim([-.05 1.05]);
xlabel('Testo pavyzdžiai, surikiuoti pagal P(1)');
ylabel('Tikimybė / tikroji klasė');
legend({'Prognozuota P(1)','Tikroji klasė','0.5 riba'},'Location','best');
title('Prognozuota 1 klasės tikimybė ir faktinė klasė');
saveas(f,fullfile(outDir,'probability_vs_truth.png')); close(f);
end

function writeProtocolLock(path,protocol,cfg)
fid=fopen(path,'w','n','UTF-8'); assert(fid~=-1); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'PROTOKOLAS UZRAKINTAS PRIES TESTA\n');
fprintf(fid,'Laikas: %s\n',protocol.lockedAt);
fprintf(fid,'Pagrindinis metodas: %s\n',cfg.primaryFamily);
fprintf(fid,'Hipoteze: %s\n',cfg.hypothesis);
fprintf(fid,'Testavimo imtis nenaudota hiperparametrams, preprocessing ar kalibravimui.\n');
fprintf(fid,'SVM cal slenkstis: %.3f; CS-SVM cal slenkstis: %.3f\n', ...
    protocol.thresholdSVM,protocol.thresholdCSSVM);
end

function writeRunSummary(outDir,manifest,dedup,leak,selectedCV,TC,TP,Hyp,Tthr,perf,cfg)
fid=fopen(fullfile(outDir,'RUN_SUMMARY.txt'),'w','n','UTF-8'); assert(fid~=-1);
c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'KALBOS GARSO FONEMU KLASIFIKAVIMAS - ATNAUJINTAS EKSPERIMENTAS\n\n');
fprintf(fid,'Duomenu saltinis: OpenML Phoneme ID 1489.\n');
fprintf(fid,'Pradiniu irasu: %d; po vienodu pozymiu vektoriu deduplikavimo: %d; pasalinta: %d.\n', ...
    manifest.rawN,manifest.n,dedup.removedRows);
fprintf(fid,'Nutekejimo auditas: train-test identisku pozymiu vektoriu = %d; train-cal = %d; cal-test = %d.\n\n', ...
    leak.TrainTestFeatureOverlap,leak.TrainCalFeatureOverlap,leak.CalTestFeatureOverlap);
fprintf(fid,'TRAIN CV PARINKTOS KONFIGURACIJOS\n');
for k=1:height(selectedCV)
    fprintf(fid,'%s | BA %.6f | F1 %.6f\n',char(selectedCV.Configuration(k)),selectedCV.MeanBA(k),selectedCV.MeanF1(k));
end
fprintf(fid,'\nGALUTINIS PALYGINIMAS - TIK TESTAVIMO IMTIS\n');
for k=1:height(TC)
    fprintf(fid,'%s | BA %.6f | F1 %.6f | Recall1 %.6f | Accuracy %.6f | Brier %.6f | AUC %.6f\n', ...
        char(TC.Model(k)),TC.BA(k),TC.F1(k),TC.Recall1(k),TC.Accuracy(k),TP.Brier(k),TP.AUC(k));
end
fprintf(fid,'\nHIPOTEZE\n');
fprintf(fid,'Atskaitos metodas: %s; DeltaBA %.6f; 95%% CI [%.6f; %.6f]; DeltaRecall1 %.6f; patvirtinta=%d\n', ...
    char(Hyp.Baseline),Hyp.DeltaBA,Hyp.CI95Low,Hyp.CI95High,Hyp.DeltaRecall1,Hyp.H1Confirmed);
fprintf(fid,'\nKALIBRAVIMO SLENKSCIU DIAGNOSTIKA\n');
for k=1:height(Tthr)
    fprintf(fid,'%s t=%.3f BA=%.6f F1=%.6f\n',char(Tthr.Model(k)),Tthr.Threshold(k),Tthr.BA(k),Tthr.F1(k));
end
fprintf(fid,'\nLaikas: %.2f s; CV: %.2f s; 1 iraso prognoze %.6g s; 1000 irasu %.6g s.\n', ...
    perf.TotalSeconds,perf.CVSeconds,perf.Predict1Seconds,perf.Predict1000Seconds);
fprintf(fid,'\nHipotezes tekstas: %s\n',cfg.hypothesis);
end

function writeAutomaticReport(outDir,manifest,dedup,leak,selectedCV,TC,TP,Hyp,Tthr,perf,cfg)
% Markdown ataskaita su realiais sio paleidimo skaiciais. Ji leidzia po kiekvieno
% pakartotinio paleidimo tureti nauja, su kodu sinchronizuota rezultatu santrauka.
fid=fopen(fullfile(outDir,'ATASKAITA_AUTOMATINE.md'),'w','n','UTF-8'); assert(fid~=-1);
c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# Kalbos garso fonemu klasifikavimo igyvendinimo rezultatai\n\n');
fprintf(fid,'**MATLAB:** %s  \n**Protokolo versija:** %s\n\n',cfg.release,cfg.version);
fprintf(fid,'## Duomenys ir nutekejimo patikra\n\n');
fprintf(fid,'Pradiniu irasu: %d. Po vienodu pozymiu vektoriu deduplikavimo: %d; pasalinta: %d. ', ...
    manifest.rawN,manifest.n,dedup.removedRows);
fprintf(fid,'Train-test identisku vektoriu po skaidymo: %d.\n\n',leak.TrainTestFeatureOverlap);
fprintf(fid,'## Train CV parinktos konfigūracijos\n\n');
fprintf(fid,'| Modelio seima | Konfiguracija | CV BA | CV F1 |\n|---|---|---:|---:|\n');
for k=1:height(selectedCV)
    fprintf(fid,'| %s | %s | %.4f | %.4f |\n',char(selectedCV.Family(k)), ...
        char(selectedCV.Configuration(k)),selectedCV.MeanBA(k),selectedCV.MeanF1(k));
end
fprintf(fid,'\n## Galutiniai rezultatai atskiroje testavimo imtyje\n\n');
fprintf(fid,'| Modelis | BA | F1 | Recall1 | Accuracy | Brier | AUC |\n|---|---:|---:|---:|---:|---:|---:|\n');
for k=1:height(TC)
    fprintf(fid,'| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n',char(TC.Model(k)), ...
        TC.BA(k),TC.F1(k),TC.Recall1(k),TC.Accuracy(k),TP.Brier(k),TP.AUC(k));
end
fprintf(fid,'\n## Hipotezes patikra\n\n');
fprintf(fid,'Stipresnis atskaitos metodas pagal train CV: **%s**. CS-RBF-SVM minus atskaitos metodas: DeltaBA=%.4f; ', ...
    char(Hyp.Baseline),Hyp.DeltaBA);
fprintf(fid,'porinio bootstrap 95%% intervalas [%.4f; %.4f], DeltaRecall1=%.4f. ', ...
    Hyp.CI95Low,Hyp.CI95High,Hyp.DeltaRecall1);
if Hyp.H1Confirmed
    fprintf(fid,'Is anksto uzrasyta H1 patvirtinta.\n');
else
    fprintf(fid,'Is anksto uzrasyta H1 nepatvirtinta.\n');
end
fprintf(fid,'\n## Tikimybiu kalibravimo diagnostika\n\n');
fprintf(fid,'| Variant | Slenkstis | BA | F1 |\n|---|---:|---:|---:|\n');
for k=1:height(Tthr)
    fprintf(fid,'| %s | %.3f | %.4f | %.4f |\n',char(Tthr.Model(k)),Tthr.Threshold(k),Tthr.BA(k),Tthr.F1(k));
end
fprintf(fid,'\n## Atkuriamumas\n\n');
fprintf(fid,'Bendra trukme %.1f s, CV %.1f s. Fiksuotos seed reiksmes saugomos experiment_config.mat. ', ...
    perf.TotalSeconds,perf.CVSeconds);
fprintf(fid,'Pagrindini eksperimenta galima pakartoti komanda `run_all`.\n');
end

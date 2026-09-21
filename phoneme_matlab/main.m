function main(dataFile,outDir,resumeFromCV)
% PHONEME: full protocol from the supplied individual assignment.
% MATLAB R2021a+ and Statistics and Machine Learning Toolbox.
% main                       uses bundled OpenML 1489, creates results/
% main('phoneme.arff','run1') uses an existing original ARFF file.
% This implementation has not been executed in MATLAB by its author.
if nargin<1, dataFile=fullfile(fileparts(mfilename('fullpath')),'phoneme.arff'); end
if nargin<2, outDir='results'; end
if nargin<3, resumeFromCV=false; end
assert(exist('fitcnet','file')==2,'MATLAB R2021a+ and Statistics Toolbox required.');
if resumeFromCV
    assert(isfolder(outDir),'Saved results directory not found.');
    required={'experimentConfig.mat','data_manifest.mat','splits.mat', ...
        'cv_details.mat','cv_results.csv','phoneme.arff'};
    for k=1:numel(required)
        assert(isfile(fullfile(outDir,required{k})),'Missing checkpoint: %s',required{k});
    end
    blocked={'frozen_models.mat','model.mat','test_predictions.csv','metrics.csv'};
    for k=1:numel(blocked)
        assert(~isfile(fullfile(outDir,blocked{k})), ...
            'Resume is only for failure before final training/test; found %s.',blocked{k});
    end
else
    assert(~isfolder(outDir),'Output directory already exists. Preserve the original test run.');
    mkdir(outDir);
end
diary(fullfile(outDir,'run_log.txt'));
cleanup=onCleanup(@() diary('off')); %#ok<NASGU>
started=tic;
if resumeFromCV
    saved=load(fullfile(outDir,'experimentConfig.mat'),'cfg'); cfg=saved.cfg;
    assert(strcmp(cfg.release,version('-release')), ...
        'Resume in the same MATLAB release that created the CV checkpoint.');
    saved=load(fullfile(outDir,'data_manifest.mat'),'manifest'); manifest=saved.manifest;
    saved=load(fullfile(outDir,'splits.mat'),'splits','cv'); splits=saved.splits; cv=saved.cv;
    saved=load(fullfile(outDir,'cv_details.mat'),'configs','cvDetails','cvSeconds');
    configs=saved.configs; cvDetails=saved.cvDetails; cvSeconds=saved.cvSeconds;
    source=fullfile(outDir,'phoneme.arff');
    assert(strcmp(fileread(source),fileread(dataFile)), ...
        'Input data differs from the saved experiment source.');
    [X,y,checkManifest]=loadData(source,outDir);
    assert(isequal(manifest.featureNames,checkManifest.featureNames));
    assert(isequal(manifest.classCounts,checkManifest.classCounts));
    Xt=X(splits.train,:); yt=y(splits.train);
    assert(isequal(sort([splits.train;splits.cal;splits.test]),(1:numel(y))'));
    for f=1:cfg.folds
        assert(isequal(test(cv,f),splits.fold==f),'Saved CV partition mismatch.');
    end
    CV=readtable(fullfile(outDir,'cv_results.csv'),'TextType','string');
    cvBA=cellfun(@(v) mean(v(:)),cvDetails(:));
    assert(height(CV)==numel(configs)&&all(abs(CV.MeanBA-cvBA)<1e-10));
    families=cellfun(@(c) c.family,configs,'UniformOutput',false);
    best=zeros(4,1);
    repair=struct('version','structure-storage-fix-v1','resumedAt',char(datetime('now')), ...
        'checkpointCVSeconds',cvSeconds,'testNotPreviouslyOpened',true);
    save(fullfile(outDir,'resume_repair.mat'),'repair');
    fprintf('RESUME: reused all %d configurations and original splits; no CV retraining.\n',numel(configs));
else
cfg=struct('seed',42,'folds',5,'mlpSeeds',[11 22 33], ...
    'bootstrapSeed',4242,'bootstrapN',2000,'noise',[0 .1 .25 .5], ...
    'noiseSeeds',101:110,'threshold',.5,'tieTolerance',.001, ...
    'iterationLimit',1000,'datasetID',1489,'version','phoneme-v1.1');
cfg.matlab=version; cfg.release=version('-release'); cfg.toolboxes=ver;
cfg.inputFile=char(dataFile);
% Decisions below are fixed before opening the held-out test.
cfg.mlpFinalRule='Report all three seeds; do not select a winning seed.';
cfg.duplicatesRule='Retain original rows; audit overlaps and limit interpretation.';
cfg.ablationRule='Fixed selected SVM parameters; CV only, no model reselection.';
save(fullfile(outDir,'experimentConfig.mat'),'cfg');
[X,y,manifest]=loadData(dataFile,outDir);
save(fullfile(outDir,'data_manifest.mat'),'manifest');
rng(cfg.seed,'twister'); pt=cvpartition(y,'HoldOut',.20);
splits.test=find(test(pt)); remaining=find(training(pt));
pc=cvpartition(y(remaining),'HoldOut',.25);
splits.train=remaining(training(pc)); splits.cal=remaining(test(pc));
assert(isempty(intersect(splits.train,splits.test)));
assert(isempty(intersect(splits.train,splits.cal)));
assert(numel(unique([splits.train;splits.cal;splits.test]))==numel(y));
cv=cvpartition(y(splits.train),'KFold',cfg.folds);
splits.fold=zeros(numel(splits.train),1);
for f=1:cfg.folds, splits.fold(test(cv,f))=f; end
save(fullfile(outDir,'splits.mat'),'splits','cv');
part=["train";"cal";"test"]; N=zeros(3,1); N0=N; N1=N;
for k=1:3
    ids=splits.(char(part(k))); N(k)=numel(ids);
    N0(k)=sum(y(ids)==0); N1(k)=sum(y(ids)==1);
end
writetable(table(part,N,N0,N1),fullfile(outDir,'split_counts.csv'));
manifest.duplicateOverlaps=duplicateAudit(X,splits);
save(fullfile(outDir,'data_manifest.mat'),'manifest');
Xt=X(splits.train,:); yt=y(splits.train);
[configs,labels,families]=makeGrid();
R=struct([]); best=zeros(4,1); cvBA=nan(numel(configs),1);
cvTimer=tic;
for q=1:numel(configs)
    c=configs{q}; seeds=cfg.seed;
    if strcmp(c.family,'MLP'), seeds=cfg.mlpSeeds; end
    scores=zeros(cfg.folds,numel(seeds));
    for f=1:cfg.folds
        a=training(cv,f); b=test(cv,f);
        pp=fitPrep(Xt(a,:),true);
        Za=applyPrep(Xt(a,:),pp); Zb=applyPrep(Xt(b,:),pp);
        for r=1:numel(seeds)
            rng(seeds(r),'twister'); m=trainOne(c,Za,yt(a),cfg);
            pred=classPrediction(m,Zb,c.family);
            v=metrics(yt(b),pred,[]); scores(f,r)=v.BA;
        end
    end
    cvBA(q)=mean(scores(:));
    R(q).Configuration=string(labels{q}); R(q).Family=string(c.family);
    R(q).MeanBA=cvBA(q); R(q).StdAcrossFolds=std(mean(scores,2));
    R(q).StdAcrossFoldSeed=std(scores(:));
    fprintf('CV %d/%d %s BA=%.5f\n',q,numel(configs),labels{q},cvBA(q));
    cvDetails{q}=scores; %#ok<AGROW>
end
cvSeconds=toc(cvTimer); CV=struct2table(R);
writetable(CV,fullfile(outDir,'cv_results.csv'));
save(fullfile(outDir,'cv_details.mat'),'cvDetails','configs','cvSeconds');
end % fresh CV or checkpoint resume
% Configurations are already ordered from simpler to more complex.
familyOrder={'LR','kNN','SVM','MLP'};
for k=1:4
    ii=find(strcmp(families,familyOrder{k}));
    eligible=ii(cvBA(ii)>=max(cvBA(ii))-cfg.tieTolerance);
    best(k)=eligible(1);
end
baseline=1; if cvBA(best(2))>cvBA(best(1)), baseline=2; end
selected=configs(best); % LR, kNN, SVM, MLP; chosen using train CV only.
selectedCV=CV(best,:); writetable(selectedCV,fullfile(outDir,'selected_parameters.csv'));
% A1/A2, A3 and a rehearsal of the noise protocol use train folds only.
runAblations(Xt,yt,cv,selected{3},cfg,CV,outDir);
pp=fitPrep(Xt,true); Zt=applyPrep(Xt,pp);
models=cell(6,1); names=["LR";"kNN";"SVM_calibrated";"MLP_seed11";"MLP_seed22";"MLP_seed33"];
trainSeconds=zeros(6,1);
for k=1:6
    c=selected{min(k,4)}; seed=cfg.seed;
    if k>=4, seed=cfg.mlpSeeds(k-3); end
    rng(seed,'twister'); timer=tic; models{k}=trainOne(c,Zt,yt,cfg);
    trainSeconds(k)=toc(timer);
end
rawSVM=models{3}; Zcal=applyPrep(X(splits.cal,:),pp);
[models{3},calibration]=fitPosterior(compact(rawSVM),Zcal,y(splits.cal));
fprintf('Calibration transform: %s\n',calibration.Type);
% model.mat is the deployable model. Save it before computing any test metric.
bundle=struct('model',models{3},'preprocess',pp,'featureNames',{manifest.featureNames}, ...
    'classNames',[0;1],'threshold',cfg.threshold,'version',cfg.version, ...
    'calibration',calibration,'configuration',selected{3});
save(fullfile(outDir,'model.mat'),'bundle');
save(fullfile(outDir,'frozen_models.mat'),'models','rawSVM','selected','baseline','cfg','pp');
% One pre-specified evaluation, including the planned perturbation study.
Ztest=applyPrep(X(splits.test,:),pp); ytest=y(splits.test);
P=zeros(numel(ytest),6); H=P; metricRows=cell(6,1);
for k=1:6
    P(:,k)=probability(models{k},Ztest); H(:,k)=double(P(:,k)>=cfg.threshold);
    metricRows{k}=metrics(ytest,H(:,k),P(:,k));
end
M=vertcat(metricRows{:}); MT=struct2table(M); MT=addvars(MT,names,'Before',1,'NewVariableNames','Model');
writetable(MT,fullfile(outDir,'metrics.csv'));
meanMLP=varfun(@mean,MT(4:6,2:end)); stdMLP=varfun(@std,MT(4:6,2:end));
writetable(meanMLP,fullfile(outDir,'mlp_seed_mean.csv'));
writetable(stdMLP,fullfile(outDir,'mlp_seed_std.csv'));
Pr=table(splits.test,ytest,'VariableNames',{'OriginalRow','TrueClass'});
for k=1:6, Pr.(['P_' char(names(k))])=P(:,k); Pr.(['Y_' char(names(k))])=H(:,k); end
writetable(Pr,fullfile(outDir,'test_predictions.csv'));
[delta,ci,boot]=pairedBootstrap(ytest,H(:,3),H(:,baseline),cfg);
confirmed=delta>=.02 && ci(1)>0;
BT=table(names(baseline),delta,ci(1),ci(2),confirmed, ...
    'VariableNames',{'Baseline','DeltaBA','CI95Low','CI95High','H1Confirmed'});
writetable(BT,fullfile(outDir,'hypothesis.csv'));
save(fullfile(outDir,'bootstrap.mat'),'boot');
[rawLabel,rawScore]=predict(rawSVM,Ztest);
rawMetric=metrics(ytest,double(rawLabel),[]);
A4=struct2table([rawMetric;M(3)]);
A4=addvars(A4,["SVM_raw_Brier_not_applicable";"SVM_calibrated"], ...
    'Before',1,'NewVariableNames','Model');
writetable(A4,fullfile(outDir,'ablation_A4_test.csv'));
save(fullfile(outDir,'raw_svm_scores.mat'),'rawScore');
calibrationPlot(ytest,P(:,3),outDir);
noiseStudy(models,names,Ztest,ytest,cfg,outDir);
errorAnalysis(X(splits.test,:),ytest,H(:,3),P(:,3),splits.test,manifest.featureNames,outDir);
f=figure('Visible','off'); confusionchart(ytest,H(:,3));
title('RBF-SVM: test confusion matrix (0=nasal, 1=oral)');
saveas(f,fullfile(outDir,'confusion_matrix.png')); close(f);
% Warm-up, then average 100 runs. Times include preprocessing and validation.
predictPhoneme(X(splits.test(1),:),bundle);
one=X(splits.test(1),:); batch=repmat(one,1000,1);
t=tic; for r=1:100, predictPhoneme(one,bundle); end; oneSeconds=toc(t)/100;
t=tic; for r=1:100, predictPhoneme(batch,bundle); end; batchSeconds=toc(t)/100;
d=dir(fullfile(outDir,'model.mat')); w=whos('models','X','P','H','Zt','Ztest');
perf=struct('TotalSeconds',toc(started),'CVSeconds',cvSeconds, ...
    'FinalTrainSeconds',trainSeconds,'Predict1Seconds',oneSeconds, ...
    'Predict1000Seconds',batchSeconds,'ModelFileBytes',d.bytes, ...
    'ResumedFromCV',resumeFromCV,'TrackedVariableBytes',sum([w.bytes]),'MemoryNote','Selected variables; not peak process memory.');
if resumeFromCV
    perf.TimingNote='TotalSeconds covers only the resumed invocation; CVSeconds is from the original invocation.';
else
    perf.TimingNote='TotalSeconds covers this invocation through metric/plot generation.';
end
save(fullfile(outDir,'performance.mat'),'perf');
writeSummary(outDir,MT,BT,calibration,manifest,perf,selectedCV);
try
    makeReport(outDir);
catch reportError
    warning('PHONEME:Report','Results were saved, but Word export failed: %s. Run makeReport again after fixing it; do not retrain.',reportError.message);
end
disp(MT); disp(BT); fprintf('Outputs saved in %s\n',outDir);
end

function [X,y,m]=loadData(path,outDir)
m=struct('datasetID',1489,'source','https://www.openml.org/d/1489', ...
    'acquiredAt',char(datetime('now','TimeZone','UTC')));
if isempty(path)
    meta=webread('https://www.openml.org/api/v1/json/data/1489');
    d=meta.data_set_description; assert(strcmp(char(string(d.id)),'1489'));
    m.openml=d; path=fullfile(outDir,'phoneme.arff');
    websave(path,char(d.url));
else
    assert(isfile(path),'Data file not found.');
end
[~,~,ext]=fileparts(path);
assert(strcmpi(ext,'.arff'),'Use the original OpenML 1489 ARFF file.');
text=fileread(path); lines=splitlines(string(text)); names={}; dataStart=[];
for i=1:numel(lines)
    line=strtrim(char(lines(i)));
    if isempty(line)||startsWith(line,'%'), continue; end
    if startsWith(lower(line),'@attribute')
        tok=regexp(line,'(?i)^@attribute\s+("[^"]+"|''[^'']+''|\S+)\s+(.+)$','tokens','once');
        assert(~isempty(tok),'Unrecognized ARFF attribute.');
        names{end+1}=regexprep(tok{1},'^[''"]|[''"]$',''); %#ok<AGROW>
    elseif strcmpi(line,'@data'), dataStart=i+1; break
    end
end
assert(~isempty(dataStart)&&numel(names)==6,'Expected six ARFF columns.');
target=find(strcmpi(names,'Class'));
if isfield(m,'openml') && isfield(m.openml,'default_target_attribute')
    target=find(strcmp(names,char(m.openml.default_target_attribute)));
end
assert(numel(target)==1,'Cannot identify Class column.');
rows=nan(numel(lines)-dataStart+1,6); n=0;
for i=dataStart:numel(lines)
    line=strtrim(char(lines(i)));
    if isempty(line)||startsWith(line,'%'), continue; end
    assert(~startsWith(line,'{'),'Sparse ARFF is not supported.');
    fields=strsplit(line,',','CollapseDelimiters',false);
    assert(numel(fields)==6,'Incorrect number of fields on ARFF line %d.',i);
    n=n+1;
    for j=1:6
        token=regexprep(strtrim(fields{j}),'^[''"]|[''"]$','');
        if strcmp(token,'?'), rows(n,j)=NaN;
        else
            rows(n,j)=str2double(token);
            assert(isfinite(rows(n,j)),'Invalid numeric field on line %d.',i);
        end
    end
end
rows=rows(1:n,:); features=setdiff(1:6,target,'stable'); X=rows(:,features); y=rows(:,target);
m.originalLabels=unique(y)';
if isequal(unique(y),[1;2])
    % OpenML 1489 v1 stores KEEL class 0/1 as ARFF nominal values 1/2.
    % Explicit version-specific mapping; never inferred by majority voting.
    m.labelMapping=[1 0;2 1]; y=y-1;
elseif isequal(unique(y),[0;1])
    m.labelMapping=[0 0;1 1];
else
    error('Unknown labels; refusing to guess the class mapping.');
end
assert(all(ismember(y,[0 1]))&&numel(unique(y))==2);
assert(size(X,1)==5404,'Unexpected dataset size; review source before modelling.');
assert(sum(y==0)==3818 && sum(y==1)==1586,'Unexpected class counts; review version and labels.');
m.featureNames=names(features); m.classSemantics={'0=nasal','1=oral'};
m.semanticSource='OpenML 1489 v1 and KEEL: stored 1 -> nasal 0, stored 2 -> oral 1; explicit mapping.';
m.n=n; m.classCounts=[sum(y==0),sum(y==1)]; m.missingPerFeature=sum(isnan(X),1);
m.duplicateRows=n-size(unique([X y],'rows'),1);
m.duplicateFeatures=n-size(unique(X,'rows'),1);
m.sourceFile=path;
% Retain exact source bytes even if an input file was supplied locally.
if ~strcmp(path,fullfile(outDir,'phoneme.arff')), copyfile(path,fullfile(outDir,'phoneme.arff')); end
fprintf('Data: %d rows, class counts %d/%d, repeated feature rows %d.\n',n,m.classCounts,m.duplicateFeatures);
end

function a=duplicateAudit(X,s)
pairs={'train','cal';'train','test';'cal','test'}; a=struct;
for k=1:3
    key=[pairs{k,1} '_' pairs{k,2}];
    a.(key)=size(intersect(X(s.(pairs{k,1}),:),X(s.(pairs{k,2}),:),'rows'),1);
    if a.(key)>0
        warning('PHONEME:Duplicates','Identical feature vectors across %s: %d. Interpret test cautiously.',key,a.(key));
    end
end
end

function pp=fitPrep(X,standardize)
pp.median=median(X,1,'omitnan');
assert(all(isfinite(pp.median)),'An entire training feature is missing.');
for j=1:size(X,2), X(isnan(X(:,j)),j)=pp.median(j); end
pp.mu=mean(X,1); pp.sigma=std(X,0,1); pp.sigma(pp.sigma==0)=1;
pp.standardize=standardize;
end

function Z=applyPrep(X,pp)
assert(~any(isinf(X(:))),'Infinite input value.');
for j=1:size(X,2), X(isnan(X(:,j)),j)=pp.median(j); end
Z=X; if pp.standardize, Z=(X-pp.mu)./pp.sigma; end
assert(all(isfinite(Z(:))),'Nonfinite preprocessed features.');
end

function [C,L,F]=makeGrid()
C={}; L={}; F={};
for a=[1 .01 .0001]
    c=struct('family','LR','lambda',a); add(c,sprintf('LR lambda=%g',a));
end
for k=[21 11 5 3]
    for w={'equal','inverse'}
        c=struct('family','kNN','k',k,'weight',w{1}); add(c,sprintf('kNN k=%d %s',k,w{1}));
    end
end
for box=[.1 1 10]
    for g=[.01 .1 1 10]
        for prior={'empirical','uniform'}
            c=struct('family','SVM','C',box,'gamma',g,'prior',prior{1});
            add(c,sprintf('SVM C=%g gamma=%g prior=%s',box,g,prior{1}));
        end
    end
end
for layers={16,32,[32 16]}
    for a=[.01 .0001]
        c=struct('family','MLP','layers',layers{1},'lambda',a);
        add(c,sprintf('MLP layers=%s lambda=%g',mat2str(layers{1}),a));
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
    case 'MLP'
        m=fitcnet(X,y,'LayerSizes',c.layers,'Activations','relu', ...
            'Lambda',c.lambda,'Standardize',false,'ClassNames',[0;1], ...
            'IterationLimit',cfg.iterationLimit);
    otherwise, error('Unknown model family.');
end
end

function h=classPrediction(m,X,family)
if strcmp(family,'SVM'), h=double(predict(m,X));
else, h=double(probability(m,X)>=.5); end
end

function p=probability(m,X)
[~,s]=predict(m,X); idx=find(m.ClassNames==1);
assert(numel(idx)==1,'Positive class not found.'); p=s(:,idx);
assert(all(isfinite(p))&&all(p>=-1e-12 & p<=1+1e-12),'Scores are not probabilities.');
p=min(1,max(0,p));
end

function m=metrics(y,h,p)
assert(numel(y)==numel(h)&&all(ismember(h,[0 1])));
tp=sum(y==1 & h==1); tn=sum(y==0 & h==0); fp=sum(y==0 & h==1); fn=sum(y==1 & h==0);
rec1=divide(tp,tp+fn); rec0=divide(tn,tn+fp);
m=struct('TN',tn,'FP',fp,'FN',fn,'TP',tp,'Accuracy',(tp+tn)/numel(y), ...
    'Recall0',rec0,'Recall1',rec1,'Precision1',divide(tp,tp+fp), ...
    'BA',(rec0+rec1)/2,'F1',divide(2*tp,2*tp+fp+fn),'Brier',NaN, ...
    'ZeroDenominator',any([tp+fn,tn+fp,tp+fp,2*tp+fp+fn]==0));
if ~isempty(p), m.Brier=mean((p-y).^2); end
end

function z=divide(a,b)
if b==0, z=0; else, z=a/b; end
end

function [delta,ci,boot]=pairedBootstrap(y,a,b,cfg)
ma=metrics(y,a,[]); mb=metrics(y,b,[]); delta=ma.BA-mb.BA;
i0=find(y==0); i1=find(y==1); rng(cfg.bootstrapSeed,'twister'); boot=zeros(cfg.bootstrapN,1);
for r=1:cfg.bootstrapN
    ids=[i0(randi(numel(i0),numel(i0),1));i1(randi(numel(i1),numel(i1),1))];
    ma=metrics(y(ids),a(ids),[]); mb=metrics(y(ids),b(ids),[]); boot(r)=ma.BA-mb.BA;
end
ci=prctile(boot,[2.5 97.5]);
end

function runAblations(X,y,cv,c,cfg,CV,outDir)
labels=["control";"A1_no_standardization";"A2_drop_1";"A2_drop_2";"A2_drop_3";"A2_drop_4";"A2_drop_5"];
values=zeros(7,cfg.folds);
noiseRows=zeros(cfg.folds*numel(cfg.noise)*numel(cfg.noiseSeeds),5); n=0;
for q=1:7
    keep=1:5; if q>=3, keep(q-2)=[]; end
    for f=1:cfg.folds
        a=training(cv,f); b=test(cv,f); pp=fitPrep(X(a,keep),q~=2);
        Za=applyPrep(X(a,keep),pp); Zb=applyPrep(X(b,keep),pp);
        rng(cfg.seed,'twister'); m=trainOne(c,Za,y(a),cfg);
        v=metrics(y(b),double(predict(m,Zb)),[]); values(q,f)=v.BA;
        if q==1
            for level=cfg.noise
                for seed=cfg.noiseSeeds
                    rng(seed,'twister'); Zn=Zb+level*randn(size(Zb));
                    vn=metrics(y(b),double(predict(m,Zn)),[]); n=n+1;
                    noiseRows(n,:)=[f,level,seed,vn.BA,vn.F1];
                end
            end
        end
    end
end
T=table(labels,mean(values,2),std(values,0,2),mean(values(1,:))-mean(values,2), ...
    'VariableNames',{'Variant','MeanBA','StdBA','DropFromControl'});
writetable(T,fullfile(outDir,'ablation_A1_A2_cv.csv'));
writetable(CV(CV.Family=="SVM",:),fullfile(outDir,'ablation_A3_grid_cv.csv'));
noiseTable=array2table(noiseRows(1:n,:),'VariableNames',{'Fold','Sigma','Seed','BA','F1'});
writetable(noiseTable,fullfile(outDir,'noise_cv_rehearsal.csv'));
end

function calibrationPlot(y,p,outDir)
bin=min(floor(p*10)+1,10); rows=cell(10,1);
for b=1:10
    use=bin==b;
    rows{b}=struct('Bin',b,'Lower',(b-1)/10,'Upper',b/10,'Count',sum(use), ...
        'MeanProbability',mean(p(use)),'ObservedFraction',mean(y(use))); %#ok<AGROW>
end
T=struct2table(vertcat(rows{:})); writetable(T,fullfile(outDir,'calibration_bins.csv'));
f=figure('Visible','off'); plot([0 1],[0 1],'k--'); hold on;
ok=T.Count>0; plot(T.MeanProbability(ok),T.ObservedFraction(ok),'o-','LineWidth',1.5);
xlabel('Mean predicted P(class 1)'); ylabel('Observed fraction of class 1');
axis([0 1 0 1]); grid on; title('SVM calibration: 10 equal-width bins');
saveas(f,fullfile(outDir,'calibration.png')); close(f);
end

function noiseStudy(models,names,Z,y,cfg,outDir)
rows=cell(numel(models)*numel(cfg.noise)*numel(cfg.noiseSeeds),1);
summary=cell(numel(models)*numel(cfg.noise),1); n=0; j=0;
for k=1:numel(models)
    clean=metrics(y,double(probability(models{k},Z)>=cfg.threshold),[]);
    for level=cfg.noise
        ba=zeros(10,1); f1=ba;
        for r=1:numel(cfg.noiseSeeds)
            rng(cfg.noiseSeeds(r),'twister'); Zn=Z+level*randn(size(Z));
            v=metrics(y,double(probability(models{k},Zn)>=cfg.threshold),[]);
            ba(r)=v.BA; f1(r)=v.F1; n=n+1;
            rows{n}=struct('Model',names(k),'Sigma',level,'Seed',cfg.noiseSeeds(r),'BA',v.BA,'F1',v.F1); %#ok<AGROW>
        end
        j=j+1; summary{j}=struct('Model',names(k),'Sigma',level,'MeanBA',mean(ba), ...
            'StdBA',std(ba),'MeanF1',mean(f1),'StdF1',std(f1), ...
            'MeanBADrop',clean.BA-mean(ba),'MeanF1Drop',clean.F1-mean(f1)); %#ok<AGROW>
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(outDir,'noise_repetitions.csv'));
T=struct2table(vertcat(summary{:})); writetable(T,fullfile(outDir,'noise_summary.csv'));
f=figure('Visible','off'); hold on;
for k=1:numel(models)
    use=T.Model==names(k); errorbar(T.Sigma(use),T.MeanBA(use),T.StdBA(use),'o-','DisplayName',names(k));
end
xlabel('Noise standard deviation'); ylabel('Balanced accuracy'); grid on;
legend('Location','best','Interpreter','none'); saveas(f,fullfile(outDir,'noise.png')); close(f);
end

function errorAnalysis(X,y,h,p,ids,featureNames,outDir)
kind=repmat("TN",numel(y),1); kind(y==1 & h==1)="TP";
kind(y==0 & h==1)="FP"; kind(y==1 & h==0)="FN";
wrong=h~=y; near=abs(p-.5)<=.05;
high=wrong & ((h==1 & p>.9)|(h==0 & p<.1));
T=table(ids,y,h,p,kind,near,high,'VariableNames', ...
    {'OriginalRow','TrueClass','PredictedClass','Probability1','Outcome','NearThreshold','HighConfidenceError'});
for j=1:5, T.(sprintf('Feature%d',j))=X(:,j); end
writetable(T,fullfile(outDir,'error_analysis_all.csv'));
writetable(T(wrong|near,:),fullfile(outDir,'error_examples.csv'));
rows=cell(20,1); r=0;
for g=["TN","FP","FN","TP"]
    use=kind==g;
    for j=1:5
        r=r+1; v=X(use,j);
        rows{r}=struct('Outcome',g,'Feature',string(featureNames{j}),'Count',numel(v), ...
            'Mean',mean(v,'omitnan'),'Median',median(v,'omitnan'),'Std',std(v,'omitnan')); %#ok<AGROW>
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(outDir,'error_feature_statistics.csv'));
f=figure('Visible','off','Position',[100 100 1000 700]);
for j=1:5
    subplot(2,3,j); boxplot(X(:,j),cellstr(kind)); title(featureNames{j},'Interpreter','none');
end
saveas(f,fullfile(outDir,'error_features.png')); close(f);
end

function writeSummary(outDir,M,B,cal,manifest,perf,selected)
fid=fopen(fullfile(outDir,'rezultatai.txt'),'w','n','UTF-8'); assert(fid~=-1);
c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'KALBOS GARSO FONEMU KLASIFIKAVIMAS: MATLAB REZULTATAI\n\n');
fprintf(fid,'Duomenu: %d; 0 klase: %d; 1 klase: %d.\n',manifest.n,manifest.classCounts);
fprintf(fid,'Tik train CV parinktos konfiguracijos:\n');
for k=1:height(selected), fprintf(fid,'%s: CV BA %.6f\n',char(selected.Configuration(k)),selected.MeanBA(k)); end
fprintf(fid,'\nTestas, slenkstis 0.5:\n');
for k=1:height(M)
    fprintf(fid,'%s: BA=%.6f, F1=%.6f, Brier=%.6f, TN=%d FP=%d FN=%d TP=%d\n', ...
        char(M.Model(k)),M.BA(k),M.F1(k),M.Brier(k),M.TN(k),M.FP(k),M.FN(k),M.TP(k));
end
fprintf(fid,'\nStipresnis baseline pagal CV: %s.\n',char(B.Baseline));
fprintf(fid,'Delta BA=%.6f; porinio bootstrap 95%% intervalas [%.6f; %.6f].\n',B.DeltaBA,B.CI95Low,B.CI95High);
if B.H1Confirmed, conclusion='H1 patvirtinta pagal is anksto nustatyta taisykle.';
else, conclusion='H1 nepatvirtinta: neivykdytas bent vienas butinas kriterijus.'; end
fprintf(fid,'%s\nKalibravimo transformacija: %s.\n',conclusion,cal.Type);
fprintf(fid,'Sio paleidimo trukme iki suvestines: %.2f s; modelio failas: %d baitu.\n',perf.TotalSeconds,perf.ModelFileBytes);
fprintf(fid,'Vieno ir 1000 irasu prognozes: %.6g s ir %.6g s.\n',perf.Predict1Seconds,perf.Predict1000Seconds);
fprintf(fid,'MLP pateiktas trimis seklomis; ju vidurkiai ir sklaida atskiruose CSV.\n');
fprintf(fid,'Triuksmo tyrimas: noise_summary.csv; abliacijos: ablation_*.csv.\n');
fprintf(fid,'Dubliuotu pozymiu sutapimai train/cal=%d, train/test=%d, cal/test=%d.\n', ...
    manifest.duplicateOverlaps.train_cal,manifest.duplicateOverlaps.train_test,manifest.duplicateOverlaps.cal_test);
fprintf(fid,'Kalbetoju ir irasu ID nezinomi; bootstrap neapima paslepto priklausomumo.\n');
fprintf(fid,'Sintetinis pozymiu triuksmas neatstoja realaus mikrofono triuksmo tyrimo.\n');
fid2=fopen(fullfile(outDir,'model_card.md'),'w','n','UTF-8'); assert(fid2~=-1);
fprintf(fid2,'# Phoneme RBF-SVM\n\nInput: five numeric acoustic features, original ARFF order.\n');
fprintf(fid2,'Feature names: %s.\n\nClass 0: nasal; class 1: oral. Threshold: 0.5.\n',strjoin(manifest.featureNames,', '));
fprintf(fid2,'Training/calibration/test: stratified 60/20/20; rng(42); five-fold train CV.\n');
fprintf(fid2,'Calibration: %s on independent calibration split.\n',cal.Type);
fprintf(fid2,'Test BA: %.6f; F1: %.6f; Brier: %.6f.\n',M.BA(3),M.F1(3),M.Brier(3));
fprintf(fid2,'Unknown speakers and recording groups; not validated for microphone audio.\n');
fprintf(fid2,'Reject nonfinite or incorrectly shaped production inputs. Training NaNs use fold-local medians.\n');
fprintf(fid2,'Source code prepared with AI assistance. Results generated only by this MATLAB run.\n');
fclose(fid2);
end

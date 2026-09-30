function result=predictPhoneme(X,modelSource)
% PREDICTPHONEME Prognozuoja nosini (0) arba burnini (1) garsa.
%
% result = predictPhoneme(X)
% result = predictPhoneme(X,'runs/run_.../model_primary.mat')
% result = predictPhoneme(X,bundle)
%
% X - N x 5 baigtiniu realiu skaiciu matrica originalia OpenML pozymiu tvarka.
% Pagrindine etikete imama is CS-RBF-SVM sprendimo. Tikimybe gaunama is to
% paties modelio, kalibruoto TIK atskiroje kalibravimo imtyje.

if nargin<2 || isempty(modelSource)
    root=fileparts(mfilename('fullpath'));
    latestFile=fullfile(root,'latest_run.mat');
    assert(isfile(latestFile),'Pirmiausia paleiskite run_all.');
    savedLatest=load(latestFile,'latestRun');
    modelSource=fullfile(savedLatest.latestRun,'model_primary.mat');
end

validateattributes(X,{'numeric'},{'2d','real','finite','nonempty','ncols',5});
if ischar(modelSource)||isstring(modelSource)
    saved=load(modelSource,'bundle'); bundle=saved.bundle;
else
    bundle=modelSource;
end

X=double(X);
Z=(X-bundle.preprocess.mu)./bundle.preprocess.sigma;
rawLabel=double(predict(bundle.rawModel,Z));
[~,score]=predict(bundle.calibratedModel,Z);
idx=find(bundle.calibratedModel.ClassNames==1);
assert(numel(idx)==1,'Modelyje nerasta 1 klase.');
p=score(:,idx);
assert(all(isfinite(p))&&all(p>=-1e-8 & p<=1+1e-8),'Netinkamos tikimybes.');
p=min(1,max(0,p));

result=struct('label',rawLabel,'probability',p, ...
    'probabilityThreshold',bundle.probabilityThreshold, ...
    'calibratedThresholdLabel',double(p>=bundle.probabilityThreshold), ...
    'version',bundle.version,'valid',true);
end

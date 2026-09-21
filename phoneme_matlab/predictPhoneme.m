function result=predictPhoneme(X,modelSource)
% X: N-by-5 finite real numeric matrix, original ARFF feature order.
% modelSource: saved model.mat path or the bundle structure.
if nargin<2, modelSource=fullfile('results','model.mat'); end
validateattributes(X,{'numeric'},{'2d','real','finite','nonempty','ncols',5});
if ischar(modelSource)||isstring(modelSource)
    saved=load(modelSource,'bundle'); bundle=saved.bundle;
else, bundle=modelSource;
end
Z=(double(X)-bundle.preprocess.mu)./bundle.preprocess.sigma;
[~,score]=predict(bundle.model,Z); idx=find(bundle.model.ClassNames==1);
assert(numel(idx)==1,'Positive class not found.'); p=score(:,idx);
assert(all(isfinite(p))&&all(p>=-1e-12 & p<=1+1e-12),'Invalid probability.');
p=min(1,max(0,p));
result=struct('label',double(p>=bundle.threshold),'probability',p, ...
    'version',bundle.version,'valid',true);
end

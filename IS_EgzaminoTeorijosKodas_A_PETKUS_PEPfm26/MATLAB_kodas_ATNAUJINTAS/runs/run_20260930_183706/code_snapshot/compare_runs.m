function comparison=compare_runs(runA,runB,tolerance)
% COMPARE_RUNS Patikrina, ar du pakartotiniai paleidimai dave tuos pacius rezultatus.
%
% Pvz.:
%   compare_runs('runs/run_20260928_210000','runs/run_20260928_211500')
%
% Didesnis nei tolerance skirtumas nepavadinamas automatiskai klaida:
% skirtingos MATLAB / BLAS versijos gali duoti labai mazus slankiojo kablelio
% skirtumus. Taciau skaidymai ir parinkti hiperparametrai turi sutapti.

if nargin<3, tolerance=1e-6; end
assert(isfolder(runA)&&isfolder(runB),'Abu paleidimo aplankai turi egzistuoti.');

A=readtable(fullfile(runA,'test_classification_metrics.csv'),'TextType','string');
B=readtable(fullfile(runB,'test_classification_metrics.csv'),'TextType','string');
PA=readtable(fullfile(runA,'selected_parameters.csv'),'TextType','string');
PB=readtable(fullfile(runB,'selected_parameters.csv'),'TextType','string');
LA=readtable(fullfile(runA,'leakage_audit.csv'),'TextType','string');
LB=readtable(fullfile(runB,'leakage_audit.csv'),'TextType','string');

assert(isequal(A.Model,B.Model),'Modeliu eiliskumas / sarasas nesutampa.');
assert(isequal(PA.Configuration,PB.Configuration),'Parinkti hiperparametrai nesutampa.');
assert(all(LA.TrainTestFeatureOverlap==0) && all(LB.TrainTestFeatureOverlap==0), ...
    'Bent viename paleidime aptiktas train-test dublikatu nutekejimas.');

numericVars={'Accuracy','Recall0','Recall1','Precision1','BA','F1'};
maxDiff=0;
for k=1:numel(numericVars)
    v=numericVars{k};
    maxDiff=max(maxDiff,max(abs(A.(v)-B.(v))));
end

comparison=struct('sameSelectedParameters',isequal(PA.Configuration,PB.Configuration), ...
    'maxClassificationMetricDifference',maxDiff, ...
    'withinTolerance',maxDiff<=tolerance,'tolerance',tolerance);

disp(A(:,{'Model','BA','F1','Recall1','Accuracy'}));
fprintf('Didziausias dvieju paleidimu metrikos skirtumas: %.3g\n',maxDiff);
fprintf('Tolerancija: %.3g; atkartojamumas pagal kriteriju: %d\n',tolerance,comparison.withinTolerance);
end

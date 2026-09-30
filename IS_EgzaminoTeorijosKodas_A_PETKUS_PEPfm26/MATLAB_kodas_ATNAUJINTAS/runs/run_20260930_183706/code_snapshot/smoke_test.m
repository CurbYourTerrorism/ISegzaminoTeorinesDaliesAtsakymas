%% GREITAS ISMOKYTO MODELIO NAUDOJIMO PATIKRINIMAS
% Pirmiausia turi buti sekmingai paleistas run_all.
root=fileparts(mfilename('fullpath'));
assert(isfile(fullfile(root,'latest_run.mat')),'Pirmiausia paleiskite run_all.');

% Techninis ivesties patikrinimas. Nuliai nera mokslinio testo dalis.
x=zeros(1,5);
r=predictPhoneme(x);
disp(r);
assert(isfield(r,'label')&&isfield(r,'probability'));
assert(ismember(r.label,[0 1]));
assert(r.probability>=0 && r.probability<=1);
fprintf('smoke_test: OK\n');

function reportPath=makeReport(outDir)
% Create a genuine .docx from saved MATLAB outputs, without Report Generator.
% Requires completed main.m results. Never trains or changes a model.
if nargin<1, outDir='results'; end
root=fileparts(mfilename('fullpath')); template=fullfile(root,'report_template.docx');
assert(isfile(template),'report_template.docx not found.');
M=readtable(fullfile(outDir,'metrics.csv'),'TextType','string');
B=readtable(fullfile(outDir,'hypothesis.csv'),'TextType','string');
S=readtable(fullfile(outDir,'selected_parameters.csv'),'TextType','string');
N=readtable(fullfile(outDir,'noise_summary.csv'),'TextType','string');
A=readtable(fullfile(outDir,'ablation_A1_A2_cv.csv'),'TextType','string');
D=load(fullfile(outDir,'data_manifest.mat'),'manifest');
C=load(fullfile(outDir,'model.mat'),'bundle');
F=load(fullfile(outDir,'experimentConfig.mat'),'cfg');
P=load(fullfile(outDir,'performance.mat'),'perf');
E=readtable(fullfile(outDir,'error_analysis_all.csv'),'TextType','string');
raw=readtable(fullfile(outDir,'ablation_A4_test.csv'),'TextType','string');
% Additional chart of actual saved test metrics.
f=figure('Visible','off','Position',[100 100 900 450]);
bar([M.BA M.F1]); ylim([0 1]); grid on;
set(gca,'XTick',1:height(M),'XTickLabel',cellstr(M.Model),'TickLabelInterpreter','none');
xtickangle(20); ylabel('Rodiklio reikšmė'); legend({'BA','F1'},'Location','southoutside','Orientation','horizontal');
title('Algoritmų rezultatai atskiroje testavimo imtyje');
saveas(f,fullfile(outDir,'model_comparison.png')); close(f);
work=tempname; mkdir(work); clean=onCleanup(@() rmdir(work,'s')); %#ok<NASGU>
unzip(template,work);
xml=fileread(fullfile(work,'word','document.xml'));
rels=fileread(fullfile(work,'word','_rels','document.xml.rels'));
media=fullfile(work,'word','media'); if ~isfolder(media), mkdir(media); end
seq=100;
put('SUMMARY',paragraph(sprintf(['Eksperimentas atliktas MATLAB %s. RBF-SVM testinis BA = %.4f, F1 = %.4f, Brier = %.4f. ' ...
    'Palyginti LR, kNN, RBF-SVM ir trys iš anksto fiksuoti MLP paleidimai.'],F.cfg.release,M.BA(3),M.F1(3),M.Brier(3))));
put('DATA',paragraph(sprintf(['Patikrinta %d įrašų ir penki požymiai. Klasės: %d nosinių ir %d burninių garsų. ' ...
    'Vienodų papildomų eilučių skaičius = %d. Požymių vektorių sutapimai tarp train ir test = %d. ' ...
    'Originalaus ARFF žymos 1 ir 2 perkoduotos į 0 ir 1.'],D.manifest.n,D.manifest.classCounts(1), ...
    D.manifest.classCounts(2),D.manifest.duplicateRows,D.manifest.duplicateOverlaps.train_test)));
put('SELECTED',makeTable({'Konfigūracija','CV BA','SD tarp dalių'}, ...
    [cellstr(S.Configuration),numCells(S.MeanBA),numCells(S.StdAcrossFolds)],[6500 1400 1700]));
put('METRICS',makeTable({'Modelis','BA','F1','Accuracy','Brier'}, ...
    [cellstr(M.Model),numCells(M.BA),numCells(M.F1),numCells(M.Accuracy),numCells(M.Brier)],[3300 1500 1500 1700 1600]));
addImage('METRICS_FIG','model_comparison.png',5.9,3.0);
put('COUNTS',makeTable({'Modelis','TN','FP','FN','TP'}, ...
    [cellstr(M.Model),numCells(M.TN,0),numCells(M.FP,0),numCells(M.FN,0),numCells(M.TP,0)],[3600 1500 1500 1500 1500]));
addImage('CONFUSION_FIG','confusion_matrix.png',4.7,3.35);
put('HYPOTHESIS',paragraph(sprintf(['Stipresnis baseline pagal mokymo CV: %s. ΔBA = %.4f; porinio bootstrap 95 %% intervalas [%.4f; %.4f]. ' ...
    'H₁ kriterijai: ΔBA ≥ 0,02 ir intervalo apatinė riba > 0. %s'],char(B.Baseline(1)),B.DeltaBA,B.CI95Low,B.CI95High,decision())));
addImage('CALIBRATION_FIG','calibration.png',5.5,3.6);
put('CALIBRATION_TEXT',paragraph(sprintf(['Kalibravimo transformacija: %s. SVM Brier = %.4f. ' ...
    'Iki kalibravimo SVM BA = %.4f, po kalibravimo BA = %.4f. ' ...
    'Neapdorotam SVM balui Brier netaikomas. Kitų metodų pirminiai įverčiai papildomai nekalibruoti.'], ...
    C.bundle.calibration.Type,M.Brier(3),raw.BA(1),raw.BA(2))));
addImage('NOISE_FIG','noise.png',5.7,3.5);
ns=N(N.Model=="SVM_calibrated",:);
put('NOISE_TABLE',makeTable({'Triukšmo σ','BA vidurkis','BA SD','F1 vidurkis','BA kritimas'}, ...
    [numCells(ns.Sigma,2),numCells(ns.MeanBA),numCells(ns.StdBA),numCells(ns.MeanF1),numCells(ns.MeanBADrop)], ...
    [1700 2200 1700 2200 1800]));
put('ABLATION_TABLE',makeTable({'Variantas','CV BA','SD','BA kritimas'}, ...
    [cellstr(A.Variant),numCells(A.MeanBA),numCells(A.StdBA),numCells(A.DropFromControl)],[4500 1700 1700 1700]));
put('ERROR_TEXT',paragraph(sprintf(['RBF-SVM: FP = %d, FN = %d; klaidingos didelio pasitikėjimo prognozės = %d; ' ...
    'įrašų, kurių tikimybė nuo slenksčio skiriasi ne daugiau kaip 0,05, = %d. ' ...
    'Originalių eilučių numeriai ir požymiai pateikti error_examples.csv. ' ...
    'Klaidų analizė yra aprašomoji ir modelio atrankos nekeičia.'],M.FP(3),M.FN(3), ...
    sum(asLogical(E.HighConfidenceError)),sum(asLogical(E.NearThreshold)))));
[~,winner]=max(M.BA);
conclusions=[paragraph(sprintf('Didžiausias atskirų testinių paleidimų BA buvo %s modelio: %.4f. Tai aprašomasis palyginimas; modelis iš naujo neparenkamas pagal testą.',char(M.Model(winner)),M.BA(winner))), ...
    paragraph(sprintf('Pagrindinis RBF-SVM, palyginti su iš anksto pasirinktu %s baseline, BA pakeitė %.4f. %s',char(B.Baseline(1)),B.DeltaBA,decision())), ...
    paragraph(sprintf('Trijų MLP sėklų vidutinis BA = %.4f, standartinis nuokrypis = %.4f. Palankiausia sėkla nenaudojama kaip vienintelis MLP rezultatas.',mean(M.BA(4:6)),std(M.BA(4:6)))), ...
    paragraph(sprintf('Esant didžiausiam σ = %.2f, SVM vidutinis BA = %.4f, pokytis nuo varianto be triukšmo = %.4f. Sintetinis triukšmas neįrodo atsparumo realiam mikrofonui.',ns.Sigma(end),ns.MeanBA(end),-ns.MeanBADrop(end))), ...
    paragraph('Kalbėtojų ir įrašų grupių ID nežinomi. Todėl išvados apribotos šiuo eilučių skaidymu, o bootstrap neapima galimos fragmentų priklausomybės.')];
put('CONCLUSIONS',conclusions);
put('PERFORMANCE',paragraph(sprintf(['Šio paleidimo laikas iki suvestinės: %.1f s; išsaugotas CV laikas: %.1f s; modelio failas: %d baitų. ' ...
    'Prognozė vienam įrašui: %.6f s; 1 000 įrašų: %.6f s. ' ...
    'Tęsiant po klaidos, šio paleidimo laikas neapima ankstesnio CV. Atmintis registruojama tik pasirinktiems kintamiesiems.'],P.perf.TotalSeconds,P.perf.CVSeconds, ...
    P.perf.ModelFileBytes,P.perf.Predict1Seconds,P.perf.Predict1000Seconds)));
assert(isempty(regexp(xml,'\{\{[A-Z_]+\}\}','once')),'Unfilled report token.');
writeText(fullfile(work,'word','document.xml'),xml);
writeText(fullfile(work,'word','_rels','document.xml.rels'),rels);
ct=fileread(fullfile(work,'[Content_Types].xml'));
if ~contains(ct,'Extension="png"')
    ct=strrep(ct,'</Types>','<Default Extension="png" ContentType="image/png"/></Types>');
end
writeText(fullfile(work,'[Content_Types].xml'),ct);
reportPath=fullfile(outDir,'Fonemu_klasifikavimo_ataskaita.docx');
zipPath=fullfile(outDir,'word_report.zip');
entries=dir(work); files={entries(~ismember({entries.name},{'.','..'})).name};
zip(zipPath,files,work); movefile(zipPath,reportPath,'f');
fprintf('Word ataskaita: %s\n',reportPath);
    function put(token,replacement)
        pattern=['<w:p(?:\s[^>]*)?>(?:(?!</w:p>).)*\{\{' token '\}\}(?:(?!</w:p>).)*</w:p>'];
        [a,b]=regexp(xml,pattern,'start','end');
        assert(numel(a)==1,'Missing or repeated token: %s',token);
        xml=[xml(1:a-1) replacement xml(b+1:end)];
    end
    function addImage(token,file,width,height)
        info=imfinfo(fullfile(outDir,file));
        scale=min(width/info.Width,height/info.Height);
        width=info.Width*scale; height=info.Height*scale;
        seq=seq+1; name=sprintf('matlab_%d.png',seq); rid=sprintf('rIdMatlab%d',seq);
        copyfile(fullfile(outDir,file),fullfile(media,name));
        rel=sprintf('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/%s"/>',rid,name);
        rels=strrep(rels,'</Relationships>',[rel '</Relationships>']);
        put(token,picture(rid,seq,width,height,file));
    end
    function d=decision()
        if asLogical(B.H1Confirmed(1)), d='H₁ patvirtinta pagal iš anksto nustatytą taisyklę.';
        else, d='H₁ nepatvirtinta: bent vienas būtinas kriterijus neįvykdytas.'; end
    end
end

function c=numCells(x,precision)
if nargin<2, precision=4; end
c=arrayfun(@(v) sprintf(['%.' num2str(precision) 'f'],v),x,'UniformOutput',false);
end
function s=esc(s)
s=char(string(s)); s=strrep(s,'&','&amp;'); s=strrep(s,'<','&lt;');
s=strrep(s,'>','&gt;'); s=strrep(s,'"','&quot;');
end
function s=paragraph(t)
s=['<w:p><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr><w:r><w:t xml:space="preserve">' esc(t) '</w:t></w:r></w:p>'];
end
function s=makeTable(headers,body,widths)
border=''; for tag={'top','left','bottom','right','insideH','insideV'}
    border=[border '<w:' tag{1} ' w:val="single" w:sz="4" w:color="D9D9D9"/>']; %#ok<AGROW>
end
s=['<w:tbl><w:tblPr><w:tblW w:w="9600" w:type="dxa"/><w:tblBorders>' border ...
    '</w:tblBorders><w:tblCellMar><w:top w:w="90" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="90" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'];
for j=1:numel(widths), s=[s sprintf('<w:gridCol w:w="%d"/>',widths(j))]; end %#ok<AGROW>
s=[s '</w:tblGrid>']; rows=[headers;body];
for i=1:size(rows,1)
    s=[s '<w:tr><w:trPr><w:cantSplit/>']; %#ok<AGROW>
    if i==1, s=[s '<w:tblHeader/>']; end %#ok<AGROW>
    s=[s '</w:trPr>']; %#ok<AGROW>
    for j=1:size(rows,2)
        shade='FFFFFF'; if i==1, shade='E6EDF2'; elseif mod(i,2)==1, shade='F7F9FB'; end
        align='center'; if j==1, align='left'; end
        bold=''; if i==1, bold='<w:b/>'; end
        s=[s sprintf('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/><w:shd w:fill="%s"/><w:vAlign w:val="center"/></w:tcPr>',widths(j),shade) ...
            '<w:p><w:pPr><w:spacing w:after="0" w:line="240"/><w:jc w:val="' align '"/></w:pPr><w:r><w:rPr>' bold ...
            '<w:sz w:val="18"/></w:rPr><w:t>' esc(rows{i,j}) '</w:t></w:r></w:p></w:tc>']; %#ok<AGROW>
    end
    s=[s '</w:tr>']; %#ok<AGROW>
end
s=[s '</w:tbl>' paragraph('')];
end
function s=picture(rid,id,width,height,description)
x=round(width*914400); y=round(height*914400);
s=sprintf(['<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>' ...
 '<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0">' ...
 '<wp:extent cx="%d" cy="%d"/><wp:docPr id="%d" name="MATLAB figure" descr="%s"/>' ...
 '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">' ...
 '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="MATLAB figure"/><pic:cNvPicPr/></pic:nvPicPr>' ...
 '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/>' ...
 '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'],x,y,id,esc(description),rid,x,y);
end
function writeText(path,s)
f=fopen(path,'w','n','UTF-8'); assert(f~=-1); c=onCleanup(@() fclose(f)); %#ok<NASGU>
fprintf(f,'%s',s);
end

function v=asLogical(x)
if islogical(x)||isnumeric(x), v=logical(x);
else, v=ismember(lower(string(x)),["true","1"]); end
end

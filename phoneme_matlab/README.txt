FONEMŲ KLASIFIKAVIMAS MATLAB APLINKOJE

Ši versija pakeičia anksčiau pateiktą kodą. Ištaisyta originalaus OpenML
ARFF klasių kodavimo klaida: faile yra 1 ir 2, programoje jos perkoduojamos
į 0 (nosinis) ir 1 (burninis). Atitiktis patikrinta visoms 5 404 eilutėms
pagal KEEL pirminį failą; patikros aprašas pridėtas JSON faile.

BŪSENA
MATLAB vykdymas rengimo aplinkoje nebuvo atliktas. Algoritmų rezultatai
nepridėti ir neišgalvoti. Word generavimo kodas taip pat turi būti patikrintas
pirmo realaus MATLAB paleidimo metu.

KAIP PALEISTI
1. Išarchyvuokite ZIP.
2. MATLAB atverkite run_project.m ir paspauskite Run.
3. Palaukite, kol baigsis modelių mokymas, įvertinimas ir Word generavimas.
4. Sukuriami:
   results/Fonemu_klasifikavimo_ataskaita.docx
   MATLAB_rezultatai.zip
5. Įkelkite MATLAB_rezultatai.zip į šį pokalbį galutinei ataskaitos patikrai.

Reikia MATLAB R2021a+ ir Statistics and Machine Learning Toolbox.
Report Generator, Microsoft Word automatizavimo, GPU ar Python nereikia.
Originalus OpenML duomenų failas pridėtas, todėl interneto nereikia.
Mokymų yra 306; trukmė priklauso nuo kompiuterio.

WORD ATASKAITA
Rezultatai skaitomi tik iš jau išsaugotų MATLAB išvesčių.
Į Word įdedami: parametrų atranka, algoritmų BA/F1/Brier lentelė,
palyginimo grafikas, sumaišties matrica, bootstrap hipotezės patikra,
kalibravimo ir triukšmo grafikai, abliacijų lentelė, klaidų suvestinė,
automatiškai pagal rezultatus suformuotos išvados bei ribotumai.
Word šablone ir ataskaitoje nėra MATLAB kodo.

Jeigu nepavyko tik Word eksportas, nepermokykite modelių. Sutvarkius klaidą:
   makeReport('results')
Taip pakartotinai naudojamos tos pačios išvestys. Report Generator nereikia.

ATKURIAMUMAS
Pirmas results aplankas negali būti perrašomas. Originalūs skaidymai,
konfigūracija, programos versijos, modeliai ir prognozės saugomi .mat ir .csv.
Mokymas: rng(42), stratifikuotas 60/20/20, train 5-fold CV.
MLP: 11, 22, 33 sėklos; visos pateikiamos, geriausia sėkla neparenkama.
SVM kalibruojamas tik cal imtyje. Pagrindinis slenkstis 0,5.
Bootstrap: 2 000 porinių stratifikuotų pakartojimų.
Triukšmas: sigma 0; 0,1; 0,25; 0,5; 10 sėklų.
Modelių įspėjimai ir konvergavimo informacija turi būti peržiūrėti.
CSV lentelėse A4 neapdoroto SVM Brier NaN reiškia „netaikoma“.

NAUDOJIMAS PO MOKYMO
   r = predictPhoneme(x, fullfile('results','model.mat'));
x yra baigtinis 1x5 skaitinis vektorius ARFF tvarka V1,V2,V3,V4,V5.
Leidžiama ir Nx5 partija. Tai akustiniai požymiai, ne mikrofono įrašas.

PAGRINDINIAI FAILAI
run_project.m          vieno paleidimo scenarijus ir rezultatų ZIP
main.m                 visas mokymas, CV, testas ir analizės
predictPhoneme.m       modelio naudojimas
makeReport.m           Word generavimas iš išsaugotų rezultatų
report_template.docx   automatinio Word eksporto šablonas
phoneme.arff           originalus OpenML 1489 v1
openml_1489_metadata.json   originalūs metaduomenys
label_mapping_verification.json   KEEL ir OpenML klasių atitikties patikra

ŠALTINIAI
https://www.openml.org/d/1489
https://sci2s.ugr.es/keel/dataset/data/classification/phoneme.zip
OpenML failo MD5: 902ff27649dbfb512f3fb01fc05c6b24

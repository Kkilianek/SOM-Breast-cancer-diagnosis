%% Kacper Kilianek (305375), Adam Piszczek (303803) [zespół nr. 22]
% Sieci neuronowe w zastosowaniach biomedycznych (SNB) – Projekt
% Projekt nr. 36: Diagnostyka raka piersi w badaniach mammograficznych za pomocą sieci SOM (katalog: Mammographic Mass_MLR)

%% ========= Przygotowanie środowiska =========

clear;
clc;
clf;
close all;

% Jeżeli nie ma folderu na wykresy, stwórz go
if ~exist("./wykresy", 'dir')
       mkdir("./wykresy")
end

%% ========= Wczytanie danych =========

% Dane wczytane są w kolejnych kolumnach BI-RADS,Age,Shape,Margin,Density,Severity
try
    M = readtable('mammographic_masses.data.txt'); % przekonwertowanie plików na txt
catch 
    fprintf("Nie udało się otworzyć pliku mammographic_masses.data.txt")
end

%% ========= Preprocessing danych (etap 1) =========

size1 = size(M,1); % zapisanie ilości wektorów cech przed preprocessingiem
M = table2array(M); % zamiana na dane numeryczne

% Proponowany sposób kodowania danych numerycznych -> single
% (single-precision number), ponieważ mamy do czynienia z liczbami 
% całkowitymi typu integer
single(M); 

% usunięcie złych danych pierwszej cechy (BI-RADS) z poza zdefiniowanego zakresu 1-5
% ustawienie warunków pilnujących podane zakresy cech:

% Pierwszy wektor cech (BI-RADS):
Cond1 = M(:,1) > 5;
Cond2 = M(:,1) < 1; 
Condition1 = Cond1 | Cond2; % połaczenie warunków

% Trzeci wektor cech (Kształt):
Cond1 = M(:,3) > 4;
Cond2 = M(:,3) < 1; 
Condition2 = Cond1 | Cond2; 

% Czwarty wektor cech (Margines):
Cond1 = M(:,4) > 5;
Cond2 = M(:,4) < 1; 
Condition3 = Cond1 | Cond2; 

% Trzeci wektor cech (Gęstość):
Cond1 = M(:,5) > 4;
Cond2 = M(:,5) < 1; 
Condition4 = Cond1 | Cond2; 

Conditions = Condition1 | Condition2 | Condition3 | Condition4;
M(Conditions,:) = []; 
size2 = size(M,1); % wielkość macierzy po redukcji błędnych wektorów
numOfDeletedRows = size1 - size2; % ilość danych, które zostały usunięte

%% ========= Preprocessing danych (etap 2) =========

% Podział danych na złośliwe i łagodne, tak aby przedstawić histogramy cech
% w klasach, które są klasyfikowane
M = sortrows(M,6);
malignant = M(:,6) == 1;
benign = M(:,6) == 0;
Malignant = M(malignant,:); % zbiór cech dla przypadku nowotworu złośliwego
Benign = M(benign,:); % zbiór cech dla przypadku nowotworu łagodnego

%% ========= Odchylenia standardowe i średnie =========

%Parametry BI-RADS
bsr=mean(M(:,1),'omitnan');
bbsr=mean(Benign(:,1),'omitnan');
bmsr=mean(Malignant(:,1),'omitnan');
bstd=std(M(:,1),'omitnan');
bbstd=std(Benign(:,1),'omitnan');
bmstd=std(Malignant(:,1),'omitnan');
%Parametry wieku
asr=mean(M(:,2),'omitnan');
absr=mean(Benign(:,2),'omitnan');
amsr=mean(Malignant(:,2),'omitnan');
astd=std(M(:,2),'omitnan');
abstd=std(Benign(:,2),'omitnan');
amstd=std(Malignant(:,2),'omitnan');
%Parametry kształtu zmiany
ssr=mean(M(:,3),'omitnan');
sbsr=mean(Benign(:,3),'omitnan');
smsr=mean(Malignant(:,3),'omitnan');
sstd=std(M(:,3),'omitnan');
sbstd=std(Benign(:,3),'omitnan');
smstd=std(Malignant(:,3),'omitnan');
%Parametry marginesu zmiany
msr=mean(M(:,4),'omitnan');
mbsr=mean(Benign(:,4),'omitnan');
mmsr=mean(Malignant(:,4),'omitnan');
mstd=std(M(:,4),'omitnan');
mbstd=std(Benign(:,4),'omitnan');
mmstd=std(Malignant(:,4),'omitnan');
%Parametry gęstości zmiany
dsr=mean(M(:,5),'omitnan');
dbsr=mean(Benign(:,5),'omitnan');
dmsr=mean(Malignant(:,5),'omitnan');
dstd=std(M(:,5),'omitnan');
dbstd=std(Benign(:,5),'omitnan');
dmstd=std(Malignant(:,5),'omitnan');
%Grupowanie zmiennych
birads=[bsr,bstd;bbsr,bbstd;bmsr,bmstd]
wiek=[asr,astd;absr,abstd;amsr,amstd]
ksztalt=[ssr,sstd;sbsr,sbstd;smsr,smstd]
margines=[msr,mstd;mbsr,mbstd;mmsr,mmstd]
gestosc=[dsr,dstd;dbsr,dbstd;dmsr,dmstd]
%% ========= Histogramy =========

figure(1)
axis tight
hold on
subplot(2,1,1)
histogram(Benign(:,1))
title('Histogramy wartości BI-RADS dla klasy łagodnej')
subplot(2,1,2)
histogram(Malignant(:,1),'FaceColor','#A2142F')
title('Histogramy wartości BI-RADS dla klasy złośliwej')
saveas(gcf,"./wykresy/histogramy_birads.png");
hold off
figure(2)
hold on
title('Histogramy wartości BI-RADS dla obu klas')
histogram(Benign(:,1))
histogram(Malignant(:,1),'FaceColor','#A2142F')
legend('Zmiana łagodna','Zmiana złośliwa')
saveas(gcf,"./wykresy/oba_histogramy_birads.png");
hold off


figure(3)
axis tight
hold on
subplot(2,1,1)
histogram(Benign(:,2))
title('Histogramy rozkładu wieku badanych dla klasy łagodnej')
subplot(2,1,2)
histogram(Malignant(:,2),'FaceColor','#A2142F')
title('Histogramy rozkładu wieku badanych dla klasy złośliwej')
saveas(gcf,"./wykresy/histogramy_age.png");
hold off
figure(4)
hold on
title('Histogramy rozkładu wieku badanych dla obu klas')
histogram(Benign(:,2))
histogram(Malignant(:,2),'FaceColor','#A2142F')
legend('Zmiana łagodna','Zmiana złośliwa')
saveas(gcf,"./wykresy/oba_histogramy_age.png");
hold off


figure(5)
axis tight
hold on
subplot(2,1,1)
histogram(Benign(:,3))
title('Histogramy kształtu masy dla klasy łagodnej')
subplot(2,1,2)
histogram(Malignant(:,3),'FaceColor','#A2142F')
title('Histogramy kształtu masy dla klasy złośliwej')
saveas(gcf,"./wykresy/histogramy_shape.png");
hold off
figure(6)
hold on
title('Histogramy kształtu masy dla obu klas')
histogram(Benign(:,3))
histogram(Malignant(:,3),'FaceColor','#A2142F')
legend('Zmiana łagodna','Zmiana złośliwa')
saveas(gcf,"./wykresy/oba_histogramy_shape.png");
hold off


figure(7)
axis tight
hold on
subplot(2,1,1)
histogram(Benign(:,4))
title('Histogramy marginesu masy dla klasy łagodnej')
subplot(2,1,2)
histogram(Malignant(:,4),'FaceColor','#A2142F')
title('Histogramy marginesu masy dla klasy złośliwej')
saveas(gcf,"./wykresy/histogramy_margin.png");
hold off
figure(8)
hold on
title('Histogramy marginesu masy dla obu klas')
histogram(Benign(:,4))
histogram(Malignant(:,4),'FaceColor','#A2142F')
legend('Zmiana łagodna','Zmiana złośliwa')
saveas(gcf,"./wykresy/oba_histogramy_margin.png");
hold off


figure(9)
axis tight
hold on
subplot(2,1,1)
histogram(Benign(:,5))
title('Histogramy gęstości masy dla klasy łagodnej')
subplot(2,1,2)
histogram(Malignant(:,5),'FaceColor','#A2142F')
title('Histogramy gęstości masy dla klasy złośliwej')
saveas(gcf,"./wykresy/histogramy_density.png");
hold off
figure(10)
hold on
title('Histogramy gęstości masy dla obu klas')
histogram(Benign(:,5))
histogram(Malignant(:,5),'FaceColor','#A2142F')
legend('Zmiana łagodna','Zmiana złośliwa')
saveas(gcf,"./wykresy/oba_histogramy_density.png");
hold off

figure(11)
histogram(M(:,6))
axis tight
title("Histogram przewlekłości masy")
xlabel("Przewlekłość (0 - łagodna, 1 - złośliwa)")
ylabel("Ilość")
saveas(gcf,"./wykresy/histogram_severity.png");

%% ========= Parametry statystyczne wartości cech =========

format long % zmiana wyświetlania dokładności zmiennych numerycznych

% wyświetlenie wartości średnich cech 
wartoscsredniabirads = mean(M(:,1));
wartoscsredniawieku = mean(M(:,2));
wartoscsredniaksztaltu = mean(M(:,3));
wartoscsredniamarginesumasy = mean(M(:,4));
wartoscsredniagestoscimasy = mean(M(:,5));
wartoscsredniaprzewleklosci = mean(M(:,6));

% wyświetlenie odchyleń standardowych cech  
odchyleniestandardowebirads = std(M(:,1));
odchyleniestandardowewieku = std(M(:,2));
odchyleniestandardoweksztaltu = std(M(:,3));
odchyleniestandardowemarginesumasy = std(M(:,4));
odchyleniestandardowegestoscimasy = std(M(:,5));
odchyleniestandardoweprzewleklosci = std(M(:,6));

%% ========= Preprocessing danych (etap 3) =========

% Przed podaniem danych wejściowych do sieci neuronowej musimy jeszcze
% przeprowadzić operację normalizacji, tak aby każda z cech miała
% identyczny wpływ na proces uczenia się sieci. Jeżeli nie dokonalibyśmy
% takich operacji, to jedna z cech (w naszym przypadku dotycząca wieku),
% miałaby największy wpływ na rozkład danych, przez to, że jej rozpiętość
% jest największa wynosząca od 18 do 96 lat. Dlatego też poniżej
% dokonaliśmy przekształceń tak aby każda cecha przyjmowała wartości w
% zakresie od 0 do 1.

% Zgodnie ze wzorem źródło: http://lh3.ggpht.com/_MrdHIr826C4/Sl7eVpJOYfI/AAAAAAAAB34/MCFvz1r_CZQ/s800/7.JPG
M(:,1) = (M(:,1)-min(M(:,1)))/(max(M(:,1)-min(M(:,1)))) * (1-0) + 0;
M(:,2) = (M(:,2)-min(M(:,2)))/(max(M(:,2)-min(M(:,2)))) * (1-0) + 0;
M(:,3) = (M(:,3)-min(M(:,3)))/(max(M(:,3)-min(M(:,3)))) * (1-0) + 0;
M(:,4) = (M(:,4)-min(M(:,4)))/(max(M(:,4)-min(M(:,4)))) * (1-0) + 0;
M(:,5) = (M(:,5)-min(M(:,5)))/(max(M(:,5)-min(M(:,5)))) * (1-0) + 0;

%% ========= Podział danych na zbiór uczący i testowy =========

Test = [Malignant(1:uint64(size(Malignant,1)/2),:) ; Benign(1:uint64(size(Benign,1)/2),:)]; % dane testowe
Train = [Malignant(uint64(size(Malignant,1)/2)+1:size(Malignant,1),:) ; Benign(uint64(size(Benign,1)/2)+1:size(Benign,1),:)]; % dane uczące
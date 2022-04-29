%% Kacper Kilianek (305375), Adam Piszczek (303803) [zespół nr. 22]
% Sieci neuronowe w zastosowaniach biomedycznych (SNB) – Projekt
% Projekt nr. 36: Diagnostyka raka piersi w badaniach mammograficznych za pomocą sieci SOM (katalog: Mammographic Mass_MLR)

%% ========= Przygotowanie środowiska =========

clear;
clc;
clf;
close all;

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


%% ========= Implementacja sieci SOM =========

som = SOM(3,60,10,100,0.1,0.05,20,0.05,2);
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

%% ========= Podział danych na zbiór uczący i zbior testowy =========

zbiorTestowy = [Malignant(1:uint64(size(Malignant,1)/2),:) ; Benign(1:uint64(size(Benign,1)/2),:)]; % dane zbiorTestowyowe
zbiorTreningowy = [Malignant(uint64(size(Malignant,1)/2)+1:size(Malignant,1),:) ; Benign(uint64(size(Benign,1)/2)+1:size(Benign,1),:)]; % dane uczące

%% ========= Implementacja sieci SOM =========

liczbaWierszySiatki = 100;
liczbaKolumnSiatki = 100;

% Liczba iteracji dla zbieżności
iteracja = 100;

%%=========== Ustawienie parametrów dla SOM ===================================
% Początkowy rozmiar sąsiedztwa topologicznego zwycięskiego neuronu
poczatkowyRozmiarSasiedztwa = 5;

% Stała czasowa początkowego rozmiaru sąsiedztwa topologicznego
stalaCzasowa = iteracja/log(poczatkowyRozmiarSasiedztwa);

% Początkowa szybkość uczenia się zmienna w czasie
poczatkowyWspolczynnikUczenia = 1;

% Stała czasowa dla zmiennej w czasie szybkości uczenia się
wspolczynnikNauki = iteracja;


mapaSOM = inicjalizacjaWag(liczbaWierszySiatki,liczbaKolumnSiatki,size(zbiorTreningowy,2));

zbiorTreningowyDane = zbiorTreningowy;

for t = 1:iteracja
    szerokosc = poczatkowyRozmiarSasiedztwa*exp(-t/stalaCzasowa);
    wariancjaSzerokosci = szerokosc^2;
    wskaznikNauki = poczatkowyWspolczynnikUczenia*exp(-t/wspolczynnikNauki);
    if wskaznikNauki <0.025
            wskaznikNauki = 0.1;
    end

    [dystansEuklidesowy, indeks] = euklidesowyDystans(zbiorTreningowy, mapaSOM, liczbaWierszySiatki, ...
                                            liczbaKolumnSiatki,size(zbiorTreningowy,1), size(zbiorTreningowy,2));
    [~,indeks2] = min(dystansEuklidesowy(:));
    [wygranyRzad,wygranaKolumna] = ind2sub(size(dystansEuklidesowy),indeks2);

    neighborhood = obliczNajblizszegoSasiada( liczbaWierszySiatki, liczbaKolumnSiatki, wygranyRzad, ...
                                            wygranaKolumna, wariancjaSzerokosci);
    mapaSOM = aktualizacjaWag( zbiorTreningowyDane, mapaSOM, liczbaWierszySiatki, liczbaKolumnSiatki, ...
                                size(zbiorTreningowy,2), indeks, wskaznikNauki, neighborhood);
    
    
    % Wektor wagowy neuronu
    dot = zeros(liczbaWierszySiatki*liczbaKolumnSiatki, size(zbiorTreningowy,2));
    % Macierz SOM do rysowania
    macierz = zeros(liczbaWierszySiatki*liczbaKolumnSiatki,1);
    % Macierz do usunięcia z rysunku
    macierzZapis = zeros(liczbaWierszySiatki*liczbaKolumnSiatki,1);
    indeks2 = 1;  
    hold on;
    f1 = figure(1);
    set(f1,'name',strcat('iteracja #',num2str(t)),'numbertitle','off');

    % Pobierz wektor wagowy neuronu
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki      
            dot(indeks2,:)=reshape(mapaSOM(r,c,:),1,size(zbiorTreningowy,2));
            indeks2 = indeks2 + 1;
        end
    end

    % Rysuj siatke SOM
    for r = 1:liczbaWierszySiatki
        rzad1 = 1+liczbaWierszySiatki*(r-1);
        rzad2 = r*liczbaWierszySiatki;
        wiersz1 = liczbaWierszySiatki*liczbaKolumnSiatki;

        macierz(2*r-1,1) = plot(dot(rzad1:rzad2,1),dot(rzad1:rzad2,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);
        macierz(2*r,1) = plot(dot(r:liczbaKolumnSiatki:wiersz1,1),dot(r:liczbaKolumnSiatki:wiersz1,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);

        macierzZapis(2*r-1,1) = macierz(2*r-1,1);
        macierzZapis(2*r,1) = macierz(2*r,1);

    end

    % Usuń wykres SOM z poprzedniej iteracji
    if t~=iteracja  
        for r = 1:liczbaWierszySiatki
            delete(macierzZapis(2*r-1,1));
            delete(macierzZapis(2*r,1));
            drawnow;
        end
    end
end



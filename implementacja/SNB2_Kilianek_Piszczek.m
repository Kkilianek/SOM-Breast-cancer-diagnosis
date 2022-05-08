%% Kacper Kilianek (305375), Adam Piszczek (303803) [zespół nr. 22]
% Sieci neuronowe w zastosowaniach biomedycznych (SNB) – Projekt
% Projekt nr. 36: Diagnostyka raka piersi w badaniach mammograficznych za pomocą sieci SOM (katalog: Mammographic Mass_MLR)

%% ========= Przygotowanie środowiska =========

clear;
clc;
close all;
format long

%% ========= Wczytanie danych =========

% Dane wczytane są w kolejnych kolumnach:
% BI-RADS, Age, Shape, Margin, Density, Severity
try
    M = readtable('mammographic_masses.data.txt'); % przekonwertowanie plików na txt
catch 
    fprintf("Nie udało się otworzyć pliku mammographic_masses.data.txt")
end

%% ========= Preprocessing danych (etap 1) =========

size1 = size(M,1); % zapisanie ilości wektorów cech przed preprocessingiem
M = table2array(M); % zamiana na dane numeryczne
M(any(ismissing(M),2),:) = []; % <- może jednak będzie trzeba usunac te dane :?

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

% M(:,1) = (M(:,1)-min(M(:,1)))/(max(M(:,1)-min(M(:,1)))) * (1-0) + 0;
% M(:,2) = (M(:,2)-min(M(:,2)))/(max(M(:,2)-min(M(:,2)))) * (1-0) + 0;
% M(:,3) = (M(:,3)-min(M(:,3)))/(max(M(:,3)-min(M(:,3)))) * (1-0) + 0;
% M(:,4) = (M(:,4)-min(M(:,4)))/(max(M(:,4)-min(M(:,4)))) * (1-0) + 0;
% M(:,5) = (M(:,5)-min(M(:,5)))/(max(M(:,5)-min(M(:,5)))) * (1-0) + 0;

%% ========= Preprocessing danych (etap 3) =========

M = sortrows(M,6);
malignant = M(:,6) == 1;
benign = M(:,6) == 0;
Malignant = M(malignant,:); % zbiór cech dla przypadku nowotworu złośliwego
Benign = M(benign,:); % zbiór cech dla przypadku nowotworu łagodnego

%% ========= Podział danych na zbiór uczący i zbior testowy =========

zbiorTestowy = [Malignant(1:uint64(size(Malignant,1)/2),:) ; Benign(1:uint64(size(Benign,1)/2),:)]; % dane zbiorTestowyowe
zbiorTreningowy = [Malignant(uint64(size(Malignant,1)/2)+1:size(Malignant,1),:) ; Benign(uint64(size(Benign,1)/2)+1:size(Benign,1),:)]; % dane uczące

%% ========= Implementacja sieci SOM =========

liczbaWierszySiatki = 5;
liczbaKolumnSiatki = 5;

iteracja = 100; % Odgórny limit iteracji potrzebny do zbieżności

%% =========== Ustawienie parametrów dla SOM =========
% Początkowy topologiczny rozmiar sąsiedztwa zwycięskiego neuronu
poczatkowyRozmiarSasiedztwa = 1;

% Stała czasowa początkowego rozmiaru sąsiedztwa topologicznego
% skąd jest wzór na to - i dlaczego akurat logarytm naturalny? <- ja ten
% kod tak jak mówiłem skopiowałem z neta (nie mam pojecia skad sa te wzory)
stalaCzasowa = iteracja/log(poczatkowyRozmiarSasiedztwa);

% Początkowa szybkość uczenia się zmienna w czasie
poczatkowyWspolczynnikUczenia = 1;

wspolczynnikNauki = iteracja; % Stała czasowa dla zmiennej w czasie szybkości uczenia się

mapaSOM = inicjalizacjaWag(liczbaWierszySiatki,liczbaKolumnSiatki,size(zbiorTreningowy(:,1:5),2));

rysujDane(zbiorTreningowy(:,1:5),zbiorTreningowy(:,6))

blad = zeros(iteracja,1);

%% =========== Proces uczenia sieci SOM =========

for t = 1:iteracja
    szerokosc = poczatkowyRozmiarSasiedztwa*exp(-t/stalaCzasowa); %tu też skąd ten wzorek wziąłeś? :D <- same
    wariancjaSzerokosci = szerokosc^2;
    wskaznikNauki = poczatkowyWspolczynnikUczenia*exp(-t/wspolczynnikNauki); %again <- same
    if wskaznikNauki <0.025
            wskaznikNauki = 0.1; %czemu tu już taki spadek? <- prawdopodobnie po to by nie byl ten wspolczynnik ultra maly
    end

    [dystansEntropy, indeks] = entropyDistance(zbiorTreningowy(:,1:5), mapaSOM, liczbaWierszySiatki, ...
                                            liczbaKolumnSiatki,size(zbiorTreningowy(:,1:5),1), size(zbiorTreningowy(:,1:5),2));
    [~,pomocnicza] = min(dystansEntropy(:));
    [wygranyRzad,wygranaKolumna] = ind2sub(size(dystansEntropy),pomocnicza);

    % ustalenie sasiedztwa neuronów
    neighborhood = obliczNajblizszegoSasiada(liczbaWierszySiatki, liczbaKolumnSiatki, wygranyRzad, ...
                                            wygranaKolumna, wariancjaSzerokosci);
    % aktualizacja mapy
    mapaSOM = aktualizacjaWag(zbiorTreningowy(:,1:5), mapaSOM, liczbaWierszySiatki, liczbaKolumnSiatki, ...
                                size(zbiorTreningowy(:,1:5),2), indeks, wskaznikNauki, neighborhood);
    
    % Wektor wagowy neuronu
    wektorWag = zeros(liczbaWierszySiatki*liczbaKolumnSiatki, size(zbiorTreningowy(:,1:5),2));
    % Macierz SOM do rysowania
    macierz = zeros(liczbaWierszySiatki*liczbaKolumnSiatki,1);
    % zmienna pomocnicza
    pomocnicza = 1;  
    figure(1);
    title('Mapa SOM - na podstawie wektorów wag')
    hold on;

    % Pobierz wektor wagowy neuronu
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki      
            wektorWag(pomocnicza,:) = reshape(mapaSOM(r,c,:),1,size(zbiorTreningowy(:,1:5),2));
            pomocnicza = pomocnicza + 1;
        end
    end
    
    figure(2)
    title('Proces uczenia sieci (przebieg błędu w zależności od liczby iteracji)')
    hold on;

    % Rysuj siatke SOM
    for r = 1:liczbaWierszySiatki
        rzad1 = 1+liczbaWierszySiatki*(r-1);
        rzad2 = r*liczbaWierszySiatki;
        wiersz1 = liczbaWierszySiatki*liczbaKolumnSiatki;
        blad(iteracja) = 2;
        figure(2)
        plot(blad) % <- tutaj trzeba ogarnąć jak rysować wykres funkcji błedu od iteracji
        % prawdopdoobnie trzeba wpasc na pomysl w jaki sposob robimy
        % klasyfikacje zlosliwa/lagodna. W 1 czesci napisalismy cos
        % takiego: decyzja związana z ustaleniem grupy danego wektora cech jest realizowana na podstawie
        % podobieństwa wartości zbiorów. Tworzona jest zmienna decyzyjna 𝑞, która w zależności od wartości
        % podobieństwa względem całego zbioru danych i jego 𝑁 regionów, wybiera k-tą ilość regionów i liczony
        % jest wtedy ułamek 𝑓𝑚 złośliwych regionów. Ustawiany jest próg decyzyjny 𝐶𝑓 (z zakresu od 0 do 1), a
        % przypadek dla nowotworu złośliwego egzekwowany jest w przypadku 𝑓𝑚 ≥ 𝐶𝑓 (inaczej klasyfikowana jest
        % zmiana łagodna) 
        % ((((Sam to jeszcze sprobuje przemyslec))))
        figure(1)
        macierz(2*r-1,1) = plot(wektorWag(rzad1:rzad2,1),wektorWag(rzad1:rzad2,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);
        macierz(2*r,1) = plot(wektorWag(r:liczbaKolumnSiatki:wiersz1,1),wektorWag(r:liczbaKolumnSiatki:wiersz1,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);
    end

    if t~=iteracja
        figure(1)
        clf
    end
end

%% <TUTAJ TESTOWALEM TYLKO na podstawie tego repo: https://github.com/KatarzynaRzeczyca/SOM_neural_network> 

%% ========= Kalibracja =========

% średnie wektory zmian zlosliwych i lagodnych
% sr_zlosliwy = sum(Malignant(1:uint64(size(Malignant,1)/2),1:5),'omitnan')/size(Malignant,1)/2;
% sr_lagodny = sum(Benign(1:uint64(size(Benign,1)/2),1:5),'omitnan')/size(Benign,1)/2;
% f=1;
% p=1;
% for j=1:liczbaWierszySiatki
%     for l=1:liczbaKolumnSiatki
%         d_lagodny = norm(sr_lagodny-reshape(mapaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)))
%         d_zlosliwy = norm(sr_zlosliwy-reshape(mapaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)))
%         if d_lagodny >= d_zlosliwy
%             wsp_lagodny(f,:) = [j,l];
%             f = f+1;
%         elseif d_lagodny < d_zlosliwy
%             wsp_zlosliwy(p,:) = [j,l];
%             p = p+1;
%         end
%     end
% end

kalibracja=[Malignant(1:uint64(size(Malignant,1)/2),1:5) ; Benign(1:uint64(size(Benign,1)/2),1:5)];
wspolrzedne=[0 0 ; 0 0];    %inicjalizacja wektora przechowującego współrzędne wyznaczonych neuronów
for i=1:2   %iteracja po wektorach kalibrujących
    d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki);   %macierz wartości roznicy miedzy kazdym neuronem i wektorem kalibrującym
    for j=1:liczbaWierszySiatki    %iteracja po neuronach-wiersze sieci
        for l=1:liczbaKolumnSiatki    %iteracja po neuronach-kolumny sieci
            d(j,l)=norm(kalibracja(i)-reshape(mapaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
        end
    end
    [M,I1]=min(d);
    [M,I2]=min(M);  %I2-nr. kolumny sieci; I1(I2)-nr.wiersza
    wspolrzedne(i,:)=[I1(I2),I2];
end

%% ========= Test =========

% liczbaz=0;   %liczba zdiagnozowanych zmian zlosliwych
% liczbal=0;   %liczba zdiagnozowanych zmian lagonych
% 
% for t = 1:size(zbiorTestowy,1)
%     d_test = zeros(liczbaWierszySiatki*liczbaKolumnSiatki,liczbaWierszySiatki*liczbaKolumnSiatki);
%     for j = 1:liczbaWierszySiatki
%         for l = 1:liczbaKolumnSiatki
%             d_test(j,l)=norm(zbiorTestowy(t,1:5)-reshape(mapaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
%         end
%     end
%     [~,pomocnicza] = min(dystansEntropy(:));
%     [wygranyRzad,wygranaKolumna] = ind2sub(size(dystansEntropy),pomocnicza);
%     
%     for i=1:size(wsp_lagodny,1)
%         if wsp_lagodny(i,:)==[wygranyRzad,wygranaKolumna]
%             liczbaz = liczbaz + 1;
%         end
%     end
%     for i=1:size(wsp_zlosliwy,1)
%         if wsp_zlosliwy(i,:)==[wygranyRzad,wygranaKolumna]
%             liczbal = liczbal + 1;
%         end
%     end
% end

[wt,kt]=size(zbiorTestowy);  %w-wiersze; k-kolumny
lp=0;   %liczba zdiagnozowanych patologii
lf=0;   %liczba zdiagnozowanych fizjologii
for i=1:wt   %iteracja po wektorach testujacych
    d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki);   %macierz wartości roznicy miedzy kazdym neuronem i wektorem testujacym
    for j=1:liczbaWierszySiatki    %iteracja po neuronach-wiersze sieci
        for l=1:liczbaKolumnSiatki    %iteracja po neuronach-kolumny sieci
            d(j,l)=norm(zbiorTestowy(i,1:5)-reshape(mapaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
        end
    end
    [M,I1]=min(d);
    [M,I2]=min(M);  %I2-nr. kolumny sieci; I1(I2)-nr.wiersza
    if I1(I2)<=wspolrzedne(1,1) && I2<=wspolrzedne(1,2)
        lp=lp+1;
    elseif I1(I2)>=wspolrzedne(2,1) && I2>=wspolrzedne(2,2)
        lf=lf+1;
    end
end

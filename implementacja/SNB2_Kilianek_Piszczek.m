%% Kacper Kilianek (305375), Adam Piszczek (303803) [zespół nr. 22]
% Sieci neuronowe w zastosowaniach biomedycznych (SNB) – Projekt
% Projekt nr. 36: Diagnostyka raka piersi w badaniach mammograficznych za pomocą sieci SOM (katalog: Mammographic Mass_MLR)

%% ========= Przygotowanie środowiska =========

clear;
clc;
close all;
format long
rng(1) % ustawienie ziarna generatora liczb losowych

%% ========= Wczytanie danych =========

% Dane wczytane są w kolejnych kolumnach:
% BI-RADS, Age, Shape, Margin, Density, Severity
try
    M = readtable('mammographic_masses.data.txt'); % przekonwertowanie plików na txt
catch 
    fprintf("Nie udało się otworzyć pliku mammographic_masses.data.txt")
end

%% ========= Preprocessing danych (etap 1 - usunięcie danych odstających) =========

size1 = size(M,1); % zapisanie ilości wektorów cech przed preprocessingiem
M = table2array(M); % zamiana na dane numeryczne

%zastąpienie brakujących wartości parametrem statystycznym
for i=1:5
    M(any(ismissing(M(:,i)),2),i)=median(M(:,i),'omitnan');
end

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

% Piąty wektor cech (Gęstość):
Cond1 = M(:,5) > 4;
Cond2 = M(:,5) < 1; 
Condition4 = Cond1 | Cond2; 

Conditions = Condition1 | Condition2 | Condition3 | Condition4;
M(Conditions,:) = []; % usunięcie elementów spoza zakresu
size2 = size(M,1); % wielkość macierzy po redukcji błędnych wektorów
numOfDeletedRows = size1 - size2; % ilość danych, które zostały usunięte

%% ========= Preprocessing danych (etap 2 - normalizacja danych) =========

M(:,1) = (M(:,1)-min(M(:,1)))/(max(M(:,1)-min(M(:,1)))) * (1-0) + 0;
M(:,2) = (M(:,2)-min(M(:,2)))/(max(M(:,2)-min(M(:,2)))) * (1-0) + 0;
M(:,3) = (M(:,3)-min(M(:,3)))/(max(M(:,3)-min(M(:,3)))) * (1-0) + 0;
M(:,4) = (M(:,4)-min(M(:,4)))/(max(M(:,4)-min(M(:,4)))) * (1-0) + 0;
M(:,5) = (M(:,5)-min(M(:,5)))/(max(M(:,5)-min(M(:,5)))) * (1-0) + 0;

%% ========= Preprocessing danych (etap 3 - podział na dwie klasy) =========

M = sortrows(M,6);
malignant = M(:,6) == 1;
benign = M(:,6) == 0;
Malignant = M(malignant,:); % zbiór cech dla przypadku nowotworu złośliwego
Benign = M(benign,:); % zbiór cech dla przypadku nowotworu łagodnego

%% ========= Podział danych na zbiór uczący i zbior testowy =========

zbiorTestowy = [Malignant(1:uint64(size(Malignant,1)/2),:) ; Benign(1:uint64(size(Benign,1)/2),:)]; % dane testowe
zbiorTreningowy = [Malignant(uint64(size(Malignant,1)/2)+1:size(Malignant,1),:) ; Benign(uint64(size(Benign,1)/2)+1:size(Benign,1),:)]; % dane uczące

%% ========= Implementacja sieci SOM =========

% ustalanie rozmiarów sieci
liczbaWierszySiatki = 6;
liczbaKolumnSiatki = 6;

iteracja = 100; % odgórny limit iteracji potrzebny do zbieżności

%% =========== Ustawienie parametrów dla SOM =========

poczatkowyRozmiarSasiedztwa = 2; % Początkowy topologiczny rozmiar sąsiedztwa zwycięskiego neuronu

% Stała czasowa początkowego rozmiaru sąsiedztwa topologicznego
stalaCzasowa = iteracja/log(poczatkowyRozmiarSasiedztwa);

poczatkowyWspolczynnikUczenia = 1; % Początkowa szybkość uczenia się zmienna w czasie

wspolczynnikNauki = iteracja; % Stała czasowa dla zmiennej w czasie szybkości uczenia się

siatkaSOM = inicjalizacjaWag(liczbaWierszySiatki,liczbaKolumnSiatki,size(zbiorTreningowy(:,1:5),2));

blad = zeros(iteracja,1); % wektor przechowujący obliczony błąd

%% =========== Proces uczenia sieci SOM =========

for t = 1:iteracja
    szerokosc = poczatkowyRozmiarSasiedztwa*exp(-t/stalaCzasowa);
    wariancjaSzerokosci = szerokosc^2;
    wskaznikNauki = poczatkowyWspolczynnikUczenia*exp(-t/wspolczynnikNauki);
    if wskaznikNauki <0.01 
            wskaznikNauki = 0.1; % jeśli współczynnik jest bardzo mały ustaw stałą aprobowalną wielkość
    end

    [obliczonyDystans, indeks] = najbizszyDystans(zbiorTreningowy(:,1:5), siatkaSOM, liczbaWierszySiatki, ...
                                            liczbaKolumnSiatki,size(zbiorTreningowy(:,1:5),1), size(zbiorTreningowy(:,1:5),2));
    [~,pomocnicza] = min(obliczonyDystans(:));
    [wygranyRzad,wygranaKolumna] = ind2sub(size(obliczonyDystans),pomocnicza);

    % ustalenie sasiedztwa neuronów
    otoczenie = obliczNajblizszegoSasiada(liczbaWierszySiatki, liczbaKolumnSiatki, wygranyRzad, ...
                                            wygranaKolumna, wariancjaSzerokosci);
    % aktualizacja mapy
    siatkaSOM = aktualizacjaWag(zbiorTreningowy(:,1:5), siatkaSOM, liczbaWierszySiatki, liczbaKolumnSiatki, ...
                                size(zbiorTreningowy(:,1:5),2), indeks, wskaznikNauki, otoczenie);
    
    
    wektorWag = zeros(liczbaWierszySiatki*liczbaKolumnSiatki, size(zbiorTreningowy(:,1:5),2)); % Wektor wagowy neuronu
    
    macierz = zeros(liczbaWierszySiatki*liczbaKolumnSiatki,1); % Macierz SOM do rysowania
 
    pomocnicza = 1; % zmienna pomocnicza w operacjach z wagami

    licznik = 0; % zliczający błędną klasyfikację sieci 

%     figure(1);
%     title('Mapa SOM - na podstawie wektorów wag')
%     hold on;
    
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki      
            wektorWag(pomocnicza,:) = reshape(siatkaSOM(r,c,:),1,size(zbiorTreningowy(:,1:5),2)); % Pobierz wektor wagowy neuronu
            pomocnicza = pomocnicza + 1;
        end
    end
    
    figure(2)
    title('Proces uczenia sieci (przebieg błędu w zależności od liczby iteracji)')
    hold on;
    xlabel('iteracja')
    ylabel('błąd klasyfikacji [w %]')

    % Rysuj siatke SOM
%     for r = 1:liczbaWierszySiatki
%         wiersz1 = 1+liczbaWierszySiatki*(r-1);
%         wiersz2 = r*liczbaWierszySiatki;
%         kolumna = liczbaWierszySiatki*liczbaKolumnSiatki;
%         figure(1)
%         macierz(2*r-1,1) = plot(wektorWag(wiersz1:wiersz2,1),wektorWag(wiersz1:wiersz2,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);
%         macierz(2*r,1) = plot(wektorWag(r:liczbaKolumnSiatki:kolumna,1),wektorWag(r:liczbaKolumnSiatki:kolumna,2),'--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',4);
%     end

%% =========== Kalibracja sieci SOM v2 =========  
    wspolrzedne=[0 0 ; 0 0]; % inicjalizacja wektora przechowującego współrzędne wyznaczonych neuronów
    figure(3)
    hold on
    d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki); % macierz wartości roznicy miedzy kazdym neuronem i wektorem kalibrującym
    i = 1;
    for j=1:liczbaWierszySiatki    % iteracja po neuronach-wiersze sieci
        for l=1:liczbaKolumnSiatki    % iteracja po neuronach-kolumny sieci
           d(j,l)=norm(zbiorTreningowy(i,1:5)-reshape(siatkaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
           [M,I1]=min(d);
           [C,I2]=min(M);
           wspolrzedne(i,:)=[I1(I2),I2];
           plot(wspolrzedne(1,1),wspolrzedne(1,2),'*r')
           plot(wspolrzedne(2,1),wspolrzedne(2,2),'*b')
           i = i + 1;
        end
     end
     [M,I1]=min(d);
     [D,I2]=min(M);  % I2-nr. kolumny sieci; I1(I2)-nr.wiersza
     wspolrzedne(i,:)=[I1(I2),I2];

%% =========== Test sieci SOM v2 =========      
    [wt,kt]=size(zbiorTestowy);  % w-wiersze; k-kolumny
    liczbaZlosliwych=0;   % liczba zdiagnozowanych patologii
    liczbaLagodnych=0;   % liczba zdiagnozowanych fizjologii
    for i=1:wt   % iteracja po wektorach testujacych
        d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki);   % macierz wartości roznicy miedzy kazdym neuronem i wektorem testujacym
        for j=1:liczbaWierszySiatki    % iteracja po neuronach-wiersze sieci
            for l=1:liczbaKolumnSiatki    % iteracja po neuronach-kolumny sieci
                d(j,l)=norm(zbiorTestowy(i,1:5)-reshape(siatkaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
            end
        end
        [M,I1]=min(d);
        [M,I2]=min(M); % I2-nr. kolumny sieci; I1(I2)-nr.wiersza
        if I1(I2)>=wspolrzedne(2,1) && I2>=wspolrzedne(2,2)
            liczbaLagodnych=liczbaLagodnych+1;
            if zbiorTestowy(wt,6) == 1
                licznik = licznik + 1;
            end
        else
            if zbiorTestowy(wt,6) == 0
                licznik = licznik + 1;
            end
            liczbaZlosliwych=liczbaZlosliwych+1;
        end
    end
    
    blad(t) = licznik/wt; % obliczenie stosunku liczby błednie zdiagnozowanych zmian do wszystkich testów
    figure(2)
    hold on;
    plot(t,blad(t)*100,'*b') % wykreślenie błędu
    
%     if t~=iteracja
%         figure(1)
%         clf; % wyczysczenie wykresu ilustrujacego mape som po kazdej iteracji
%     end
end


figure(2)
yline(mean(blad)*100,'--k','wartość średnia');
yline(min(blad)*100,'--b','wartość minimalna');

%% =========== Wyniki procesu uczenia sieci SOM =========

fprintf("Liczba zmian złośliwych w zbiorze testującym: " + sum(zbiorTestowy(:,6) == 1));
fprintf("\nLiczba zmian łagodynch w zbiorze testującym: " + sum(zbiorTestowy(:,6) == 0));
fprintf("\nLiczba sklasyfikowanych zmian złośliwych w ostatniej iteracji: " + liczbaZlosliwych);
fprintf("\nLiczba sklasyfikowanych zmian łagodynch w ostatniej iteracji: " + liczbaLagodnych);
fprintf("\nBłąd klasyfikacji po " + iteracja + " iteracjach wynosi: " + blad(iteracja) + "\n");

% tutaj trzeba okreslic czulosc/specyficznosc, jeszcze wypisac jakie byly
% wspolczynniki lagodne/zlosliwe, ktore byly potrzebne do klasyfikacji (w
% sensie jakie progi decyzyjne uzylismy)

figure(3)
legend('złośliwe','łagodne')
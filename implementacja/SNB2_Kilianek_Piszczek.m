%% Kacper Kilianek (305375), Adam Piszczek (303803) [zespół nr. 22]
% Sieci neuronowe w zastosowaniach biomedycznych (SNB) – Projekt
% Projekt nr. 36: Diagnostyka raka piersi w badaniach mammograficznych za pomocą sieci SOM (katalog: Mammographic Mass_MLR)

%% ========= Przygotowanie środowiska =========

clear;
clc;
close all;
format long
rng(303803)

%% ========= Wczytanie danych =========

try
    M = readtable('mammographic_masses.data.txt'); % przekonwertowanie plików na txt
catch 
    fprintf("Nie udało się otworzyć pliku mammographic_masses.data.txt")
end

%% ========= Preprocessing danych (etap 1 - usunięcie danych odstających) =========

size1 = size(M,1); % zapisanie ilości wektorów cech przed preprocessingiem
M = table2array(M); % zamiana na dane numeryczne

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

%% ========= Preprocessing danych (etap 4 - uzupełnienie niekompletnych danych) =========

%zastąpienie brakujących wartości parametrem statystycznym
for i=1:5
    Malignant(any(ismissing(Malignant(:,i)),2),i)=median(Malignant(:,i),'omitnan');
    Benign(any(ismissing(Benign(:,i)),2),i)=median(Benign(:,i),'omitnan');
end

%% ========= Podział danych na zbiór uczący i zbior testowy =========

zbiorTestowy = [Malignant(1:uint64(size(Malignant,1)/2),:) ; Benign(1:uint64(size(Benign,1)/2),:)]; % dane testowe
zbiorTreningowy = [Malignant(uint64(size(Malignant,1)/2)+1:size(Malignant,1),:) ; Benign(uint64(size(Benign,1)/2)+1:size(Benign,1),:)]; % dane uczące

%% =========== Ustawienie parametrów dla SOM =========

% ustalanie rozmiarów sieci
liczbaWierszySiatki = 6;
liczbaKolumnSiatki = 6;

iteracja = 100; % odgórny limit iteracji potrzebny do zbieżności

poczatkowyRozmiarSasiedztwa = 2; % Początkowy topologiczny rozmiar sąsiedztwa zwycięskiego neuronu

stalaCzasowa = iteracja/log(poczatkowyRozmiarSasiedztwa); % Stała czasowa początkowego rozmiaru sąsiedztwa topologicznego

poczatkowyWspolczynnikUczenia = 1; % Początkowa szybkość uczenia się zmienna w czasie

wspolczynnikNauki = iteracja; % Stała czasowa dla zmiennej w czasie szybkości uczenia się

siatkaSOM = zeros(liczbaWierszySiatki,liczbaKolumnSiatki,5); % prealokacja

    for i = 1:liczbaWierszySiatki
        for j = 1:liczbaKolumnSiatki
            siatkaSOM(i,j,:) = rand(1,5); % wylosowanie kolejno wektorów wag
        end
    end

blad = zeros(iteracja,1); % wektor przechowujący obliczony błąd w trakcie uczenia sieci

%% =========== Proces uczenia sieci SOM =========

for t = 1:iteracja
    szerokosc = poczatkowyRozmiarSasiedztwa*exp(-t/stalaCzasowa);
    wariancjaSzerokosci = szerokosc^2;
    wskaznikNauki = poczatkowyWspolczynnikUczenia*exp(-t/wspolczynnikNauki);
    if wskaznikNauki < 0.01 
            wskaznikNauki = 0.1; % jeśli współczynnik jest bardzo mały ustaw stałą aprobowalną wartość
    end

    [obliczonyDystans, indeks] = najbizszyDystans(zbiorTreningowy(:,1:5), siatkaSOM, liczbaWierszySiatki, ...
                                            liczbaKolumnSiatki,size(zbiorTreningowy(:,1:5),1), size(zbiorTreningowy(:,1:5),2));
    [~,pomocnicza] = min(obliczonyDystans(:));
    [wygranyRzad,wygranaKolumna] = ind2sub(size(obliczonyDystans),pomocnicza);

    % ustalenie sasiedztwa neuronów
    otoczenie = obliczNajblizszegoSasiada(liczbaWierszySiatki, liczbaKolumnSiatki, wygranyRzad, ...
                                            wygranaKolumna, wariancjaSzerokosci);
    % aktualizacja siatki SOM
    siatkaSOM = aktualizacjaWag(zbiorTreningowy(:,1:5), siatkaSOM, liczbaWierszySiatki, liczbaKolumnSiatki, ...
                                size(zbiorTreningowy(:,1:5),2), indeks, wskaznikNauki, otoczenie);
    
    wektorWag = zeros(liczbaWierszySiatki*liczbaKolumnSiatki, size(zbiorTreningowy(:,1:5),2)); % prealokacja wektoru wag
 
    pomocnicza = 1; % zmienna pomocnicza podczas liczenia wag
    
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki      
            wektorWag(pomocnicza,:) = reshape(siatkaSOM(r,c,:),1,size(zbiorTreningowy(:,1:5),2)); % Pobierz wektor wagowy neuronu
            pomocnicza = pomocnicza + 1;
        end
    end

    % Kalibracja sieci SOM
    wspolrzedne=zeros(size(zbiorTreningowy,1),2); % inicjalizacja wektora przechowującego współrzędne wyznaczonych neuronów 
    d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki); % macierz wartości roznicy miedzy kazdym neuronem i wektorem kalibrującym

    for i=1:size(zbiorTreningowy,1)
        for j=1:liczbaWierszySiatki
            for l=1:liczbaKolumnSiatki
               d(j,l)=norm(zbiorTreningowy(i,1:5)-reshape(siatkaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
               [~,pomocnicza] = min(d(:));
               [I1,I2] = ind2sub(size(d),pomocnicza);
               wspolrzedne(i,:)=[I1,I2];     
            end
        end
    end

    % ustalenie najczęściej zapalanych neuronów
    wspolrzednezlosliwe=unique(wspolrzedne(1:216,:),'rows');
    wspolrzednelagodne=unique(wspolrzedne(217:end,:),'rows');
    iloscz=zeros(size(wspolrzednezlosliwe,1),1);
    iloscl=zeros(size(wspolrzednelagodne,1),1);

    for i=1:216
        for j=1:size(wspolrzednezlosliwe)
            if wspolrzedne(i,1)==wspolrzednezlosliwe(j,1) && wspolrzedne(i,2)==wspolrzednezlosliwe(j,2)
                iloscz(j)=iloscz(j)+1;
            end
        end
    end

    for i=217:size(wspolrzedne,1)
        for j=1:size(wspolrzednelagodne)
            if wspolrzedne(i,1)==wspolrzednelagodne(j,1) && wspolrzedne(i,2)==wspolrzednelagodne(j,2)
                iloscl(j)=iloscl(j)+1;
            end
        end
    end

    % stworzenie heatmapy najczęściej zapalanych neuronów
    podsumz=[wspolrzednezlosliwe,iloscz];
    podsuml=[wspolrzednelagodne,iloscl];
    heatmapazlosliwa = zeros(liczbaWierszySiatki,liczbaKolumnSiatki);
    heatmapalagodna = zeros(liczbaWierszySiatki,liczbaKolumnSiatki);

    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki 
            for g = 1:size(podsumz,1)
                if r == podsumz(g,1) && c == podsumz(g,2) 
                    heatmapazlosliwa(r,c) = podsumz(g,3);
                end
            end
        end
    end
    
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki 
            for g = 1:size(podsuml,1)
                if r == podsuml(g,1) && c == podsuml(g,2) 
                    heatmapalagodna(r,c) = podsuml(g,3);
                end
            end
        end
    end

    % obliczenie końcowych wyników (najczęściej zapalanych neuronów względem klasy złośliwej)
    wynik = heatmapazlosliwa > heatmapalagodna;
    wynik = medfilt2(wynik,'symmetric'); % <- tutaj musimy się zdecydować czy to robimy czy nie
    
    % Test sieci SOM - poprawność na danych Treningowych (błąd uczenia)
    [wt,~] = size(zbiorTreningowy);
    liczbaZlosliwych = 0;
    liczbaLagodnych = 0;
    licznik = 0;
    for i = 1:wt
        d = zeros(liczbaWierszySiatki,liczbaKolumnSiatki);
        for j = 1:liczbaWierszySiatki
            for l = 1:liczbaKolumnSiatki
                % obliczenie odległości wektora cech od wektora sieci
                d(j,l) = norm(zbiorTreningowy(i,1:5)-reshape(siatkaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
            end
        end
        [~,pomocnicza] = min(d(:));
        [I1,I2] = ind2sub(size(d),pomocnicza);
        if wynik(I1,I2) == 1 
            liczbaZlosliwych = liczbaZlosliwych + 1;
            if zbiorTreningowy(i,6) == 0
                licznik = licznik + 1;
            end
        else
            liczbaLagodnych = liczbaLagodnych + 1;
            if zbiorTreningowy(i,6) == 1
                licznik = licznik + 1;
            end
        end
    end

    blad(t) = licznik/wt *100; % obliczenie błedu 
    fprintf("Iteracja: " + t + "\n");
end

figure(1)
title('Proces uczenia sieci (przebieg błędu w zależności od liczby iteracji)')
hold on;
xlabel('iteracja')
ylabel('błąd klasyfikacji [w %]')
plot(blad,'--*b')
yline(min(blad),'-k','wartość minimalna');

figure(2)
imagesc(wynik)
title('Mapa zapalanych neuronów łagodna/złośliwa klasyfikacja')

%% =========== Wyniki procesu uczenia sieci SOM =========

fprintf('==== Wyniki procesu uczenia sieci SOM ====');
fprintf("\nLiczba zmian złośliwych w zbiorze treningowym: " + sum(zbiorTreningowy(:,6) == 1));
fprintf("\nLiczba zmian łagodynch w zbiorze treningowym: " + sum(zbiorTreningowy(:,6) == 0));
fprintf("\nLiczba sklasyfikowanych zmian złośliwych w ostatniej iteracji: " + liczbaZlosliwych);
fprintf("\nLiczba sklasyfikowanych zmian łagodynch w ostatniej iteracji: " + liczbaLagodnych);
fprintf("\nBłąd klasyfikacji w procentach po " + iteracja + " iteracjach wynosi: " + blad(iteracja) + "\n");

%% =========== Test sieci SOM - na danych nieznanych (błąd po uczeniu) =========   

liczniktest = 0;
[wt,kt] = size(zbiorTestowy);
liczbaZlosliwychtest = 0;
liczbaLagodnychtest = 0;
prawdziwiedodatni = 0;
prawdziwieujemny = 0;
falszywiedodatni = 0;
falszywieujemny = 0;
for i=1:wt
    d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki);
    for j=1:liczbaWierszySiatki
        for l=1:liczbaKolumnSiatki
            d(j,l)=norm(zbiorTestowy(i,1:5)-reshape(siatkaSOM(j,l,:),1,size(zbiorTestowy(:,1:5),2)));
        end
    end
    [~,pomocnicza] = min(d(:));
    [I1,I2] = ind2sub(size(d),pomocnicza);
    if wynik(I1,I2) == 1 
        liczbaZlosliwychtest = liczbaZlosliwychtest + 1;
        if zbiorTestowy(i,6) == 0
            prawdziwieujemny = prawdziwieujemny + 1;
            liczniktest = liczniktest + 1;
        else
            prawdziwiedodatni = prawdziwiedodatni + 1;
        end
    else
        liczbaLagodnychtest = liczbaLagodnychtest + 1;
        if zbiorTestowy(i,6) == 1
            liczniktest = liczniktest + 1;
            falszywieujemny = falszywieujemny + 1;
        else
            falszywiedodatni = falszywiedodatni + 1;
        end
    end
end

bladtest = liczniktest/wt * 100;

%% =========== Wyniki testu nauczenia sieci SOM =========

fprintf('\n==== Wyniki testu nauczenia sieci SOM ====');
fprintf("\nLiczba zmian złośliwych w zbiorze testującym: " + sum(zbiorTestowy(:,6) == 1));
fprintf("\nLiczba zmian łagodynch w zbiorze testującym: " + sum(zbiorTestowy(:,6) == 0));
fprintf("\nLiczba wszystkich wykrytych zmian złośliwych: " + liczbaZlosliwychtest);
fprintf("\nLiczba wszystkich wykrytych zmian łagodnych: " + liczbaLagodnychtest);
fprintf("\nBłąd klasyfikacji ogółem w procentach: " + bladtest + "\n");

%% =========== Czułość i specyficzność sieci SOM =========

czulosc = prawdziwiedodatni / (prawdziwiedodatni + falszywieujemny);
specyficznosc = prawdziwieujemny / (prawdziwieujemny + falszywiedodatni);
fprintf('\n==== Czułość i specyficzność sieci SOM ====');
fprintf("\nCzułość: " + czulosc);
fprintf("\nSpecyficzność: " + specyficznosc + "\n");

%% =========== TODO ===========

% sprawdzic czy na pewno dobrze aktualizujemy wagi i przeprowadzamy proces
% uczenia

% decyzja czy uzywamy filtru medianowego czy nie?

% ostatnie chyba pytanie, czy przechodzimy na wektoryzacje?

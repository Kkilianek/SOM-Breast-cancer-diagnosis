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

%% =========== Kalibracja sieci SOM =========  
    wspolrzedne=zeros(size(zbiorTreningowy,1),2); % inicjalizacja wektora przechowującego współrzędne wyznaczonych neuronów
    
    d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki); % macierz wartości roznicy miedzy kazdym neuronem i wektorem kalibrującym
    % 1-216 złośliwe
    % 217-471 łagodne

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
    podsumz=[wspolrzednezlosliwe,iloscz];
    podsuml=[wspolrzednelagodne,iloscl];
    klasyflagodne=setdiff(wspolrzednelagodne,wspolrzednezlosliwe,'rows');
    klasyfzlosliwe=setdiff(wspolrzednezlosliwe,wspolrzednelagodne,'rows');
    if size(klasyflagodne,1)>=size(klasyfzlosliwe,1)
        klasyfikacja=klasyflagodne;
        wybrano="l";
        fprintf("Wybrano klasyfikację na podstawie zmian łagodnych"+newline);
    else 
        klasyfikacja=klasyfzlosliwe;
        wybrano="z";
        fprintf("Wybrano klasyfikację na podstawie zmian złośliwych"+newline);
    end


%% =========== Test sieci SOM - poprawność na danych Treningowych (błąd uczenia) =========      
    [wt,kt]=size(zbiorTreningowy);  % w-wiersze; k-kolumny
    liczbaZlosliwych=0;   % liczba zdiagnozowanych patologii
    liczbaLagodnych=0;   % liczba zdiagnozowanych fizjologii
    for i=1:wt   % iteracja po wektorach testujacych
        d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki);   % macierz wartości roznicy miedzy kazdym neuronem i wektorem testujacym
        for j=1:liczbaWierszySiatki    % iteracja po neuronach-wiersze sieci
            for l=1:liczbaKolumnSiatki    % iteracja po neuronach-kolumny sieci
                d(j,l)=norm(zbiorTreningowy(i,1:5)-reshape(siatkaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
            end
        end
        [~,pomocnicza] = min(d(:));
        [I1,I2] = ind2sub(size(d),pomocnicza);
        switch wybrano
            case "l"
                for z=1:size(klasyfikacja,1)
                    if I1==klasyfikacja(z,1) && I2==klasyfikacja(z,2)
                        liczbaLagodnych=liczbaLagodnych+1;
                        if zbiorTestowy(i,6)==1
                            licznik=licznik+1;
                        end
                    else
                        liczbaZlosliwych=liczbaZlosliwych+1;
                         if zbiorTestowy(i,6)==0
                            licznik=licznik+1;
                         end
                    end
                end
                break
                case "z"
                for z=1:size(klasyfikacja,1)
                    if I1==klasyfikacja(z,1) && I2==klasyfikacja(z,2)
                        liczbaZlosliwych=liczbaZlosliwych+1;
                        if zbiorTestowy(i,6)==0
                            licznik=licznik+1;
                        end
                    else
                        liczbaLagodnych=liczbaLagodnych+1;
                         if zbiorTestowy(i,6)==1
                            licznik=licznik+1;
                         end
                    end
                end
                break
        end
                
    end
    
    blad(t) = licznik/wt; % obliczenie stosunku liczby błednie zdiagnozowanych zmian do wszystkich testów
    figure(2)
    hold on;
    plot(t,blad(t)*100,'*b') % wykreślenie błędu
    
end
figure(2)
yline(mean(blad)*100,'--k','wartość średnia');
yline(min(blad)*100,'--b','wartość minimalna');

%% =========== Wyniki procesu uczenia sieci SOM =========

fprintf("Liczba zmian złośliwych w zbiorze treningowym: " + sum(zbiorTreningowy(:,6) == 1));
fprintf("\nLiczba zmian łagodynch w zbiorze treningowym: " + sum(zbiorTreningowy(:,6) == 0));
fprintf("\nLiczba sklasyfikowanych zmian złośliwych w ostatniej iteracji: " + liczbaZlosliwych);
fprintf("\nLiczba sklasyfikowanych zmian łagodynch w ostatniej iteracji: " + liczbaLagodnych);
fprintf("\nBłąd klasyfikacji po " + iteracja + " iteracjach wynosi: " + blad(iteracja) + "\n");

%% =========== Test sieci SOM - na danych nieznanych (błąd po uczeniu) =========      
    liczniktest=0;
    [wt,kt]=size(zbiorTestowy);  % w-wiersze; k-kolumny
    liczbaZlosliwychtest=0;   % liczba zdiagnozowanych patologii
    liczbaLagodnychtest=0;   % liczba zdiagnozowanych fizjologii
    for i=1:wt   % iteracja po wektorach testujacych
        d=zeros(liczbaWierszySiatki,liczbaKolumnSiatki);   % macierz wartości roznicy miedzy kazdym neuronem i wektorem testujacym
        for j=1:liczbaWierszySiatki    % iteracja po neuronach-wiersze sieci
            for l=1:liczbaKolumnSiatki    % iteracja po neuronach-kolumny sieci
                d(j,l)=norm(zbiorTestowy(i,1:5)-reshape(siatkaSOM(j,l,:),1,size(zbiorTreningowy(:,1:5),2)));
            end
        end
        [~,pomocnicza] = min(d(:));
        [I1,I2] = ind2sub(size(d),pomocnicza);
        switch wybrano
            case "l"
                for z=1:size(klasyfikacja,1)
                    if I1==klasyfikacja(z,1) && I2==klasyfikacja(z,2)
                        liczbaLagodnychtest=liczbaLagodnychtest+1;
                        if zbiorTestowy(i,6)==1
                            liczniktest=liczniktest+1;
                        end
                    else
                        liczbaZlosliwychtest=liczbaZlosliwychtest+1;
                         if zbiorTestowy(i,6)==0
                            liczniktest=liczniktest+1;
                         end
                    end
                end
                case "z"
                for z=1:size(klasyfikacja,1)
                    if I1==klasyfikacja(z,1) && I2==klasyfikacja(z,2)
                        liczbaZlosliwychtest=liczbaZlosliwychtest+1;
                        if zbiorTestowy(i,6)==0
                            liczniktest=liczniktest+1;
                        end
                    else
                        liczbaLagodnychtest=liczbaLagodnychtest+1;
                         if zbiorTestowy(i,6)==1
                            liczniktest=liczniktest+1;
                         end
                    end
                end
        end
                
    end
    
    bladtest = liczniktest/wt; % obliczenie stosunku liczby błednie zdiagnozowanych zmian do wszystkich testów
    


%% =========== Wyniki testu nauczenia sieci SOM =========

fprintf("Liczba zmian złośliwych w zbiorze testującym: " + sum(zbiorTestowy(:,6) == 1));
fprintf("\nLiczba zmian łagodynch w zbiorze testującym: " + sum(zbiorTestowy(:,6) == 0));
fprintf("\nLiczba wszystkich wykrytych zmian złośliwych: " + liczbaZlosliwychtest);
fprintf("\nLiczba wszystkich wykrytych zmian łagodnych: " + liczbaLagodnychtest);
fprintf("\nBłąd klasyfikacji ogółem w %:"+bladtest*100);

% tutaj trzeba okreslic czulosc/specyficznosc, jeszcze wypisac jakie byly
% wspolczynniki lagodne/zlosliwe, ktore byly potrzebne do klasyfikacji (w
% sensie jakie progi decyzyjne uzylismy)
function [obliczonyDystans, i] = najbizszyDystans(zbiorTreningowy, siatkaSOM, liczbaWierszySiatki,...
                                liczbaKolumnSiatki, iloscDanych, wymiarDanych)
% Funkcja odpowiedzialna za obliczenie najbliższego dystnasu do neuronu
% jako odległości euklidesowej.

    obliczonyDystans = zeros(liczbaWierszySiatki, liczbaKolumnSiatki); % prealokacja

    i = randi([1 iloscDanych]); % wybranie losowego indeksu wiersza ze zbioru treningowego
    
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki
            V = zbiorTreningowy(i,:) - reshape(siatkaSOM(r,c,:),1,wymiarDanych);
            obliczonyDystans(r,c) = sqrt(V*V'); % zwrócenie dystansu
        end
    end
end
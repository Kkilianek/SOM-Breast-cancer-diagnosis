function [obliczonyDystans, i] = najbizszyDystans(zbiorTreningowy, mapaSOM, liczbaWierszySiatki,...
                                liczbaKolumnSiatki, iloscDanych, wymiarDanych)
% Opis funkcji...

    obliczonyDystans = zeros(liczbaWierszySiatki, liczbaKolumnSiatki);

    i = randi([1 iloscDanych]);
    
    for r = 1:liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki
            V = zbiorTreningowy(i,:) - reshape(mapaSOM(r,c,:),1,wymiarDanych);
            obliczonyDystans(r,c) = sqrt(V*V');
        end
    end

end
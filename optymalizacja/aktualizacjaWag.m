function aktualizacjWagWektorow = aktualizacjaWag(zbiorTreningowy, siatkaSOM, liczbaWierszySiatki, ... 
                        liczbaKolumnSiatki, wymiarDanych, indeks, wspolczynnikUczenia, otoczenie)
% Funkcja odpowiedzialna za aktualizowanie wag w każdej iteracji w
% zależności od otoczenia neuronu i współczynnika uczenia sieci

    aktualizacjWagWektorow = zeros(liczbaWierszySiatki, liczbaKolumnSiatki, wymiarDanych); % prealokacja
    
    for r = 1: liczbaWierszySiatki
        for c = 1:liczbaKolumnSiatki
            % obliczenie kolejny zaktualizowanych wektorów wag
            aktualnyWektorWag = reshape(siatkaSOM(r,c,:),1,wymiarDanych);
            aktualizacjWagWektorow(r,c,:) = aktualnyWektorWag + wspolczynnikUczenia*otoczenie(r,c)*(zbiorTreningowy(indeks,:)-aktualnyWektorWag);
        end
    end
end
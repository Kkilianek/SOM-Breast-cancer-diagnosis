function macierzWag = inicjalizacjaWag(liczbaWierszy,liczbaKolumn, iloscDanych)
% Funkcja odpowiedzialna za inicjalizowanie wag (wywoływana tylko raz, przed
% procesem uczenia) 

    macierzWag = zeros(liczbaWierszy,liczbaKolumn,iloscDanych); % prealokacja

    for i = 1:liczbaWierszy
        for j = 1:liczbaKolumn
            macierzWag(i,j,:) = rand(1,iloscDanych); % wylosowanie kolejno wektorów wag
        end
    end
end


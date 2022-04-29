function wektorWag = inicjalizacjaWag(liczbaWierszy,liczbaKolumn, iloscDanych)
%INICJALIZACJAWAG Summary of this function goes here
%   Detailed explanation goes here
wektorWag = zeros(liczbaWierszy,liczbaKolumn,iloscDanych);
    for i = 1:liczbaWierszy
        for j = 1:liczbaKolumn
            wektorWag(i,j,:) = rand(1,iloscDanych);
        end
    end
end


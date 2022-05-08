function rysujDane(zbiorTreningowy, y)
% Opis funkcji...

    figure(4);
    title('Wizualizacja danych treningowych')
    hold on;
    cluster1 = find(y==0);
    cluster2 = find(y==1);
    plot(zbiorTreningowy(cluster1,1),zbiorTreningowy(cluster1,2),'yo','LineWidth',2,'MarkerSize',12);
    plot(zbiorTreningowy(cluster2,1),zbiorTreningowy(cluster2,2),'m+','LineWidth',2,'MarkerSize',12);
    hold off
    legend('zmiana łagodna','zmiana złośliwa')

end


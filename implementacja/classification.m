function result=classification(x,y,wspf,wspp) %zwraca 0 jeœli jest fizjo, a 1 jeœli pato
for i=1:size(wspf,1)
    if wspf(i,:)==[x,y]
        result=0;
    end
end
for i=1:size(wspp,1)
    if wspp(i,:)==[x,y]
        result=1;
    end
end
end